# =============================================================================
# LOCAL AI SYSTEM FOR DOCUMENT PROCESSING
# AI Engineer Technical Assessment
# =============================================================================

import os
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime

# Suppress noisy font warnings from PyPDF2
warnings.filterwarnings("ignore", message="Could not get FontBBox.*")

# PDF processing
import pdfplumber
from PyPDF2 import PdfReader

# Local ML models
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False
    print("Warning: transformers/torch not installed. Bonus QA feature disabled.")


# =============================================================================
# UTILITY: JSON Serialization
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that converts NumPy types to native Python types.
    Fixes: TypeError: Object of type float32 is not JSON serializable
    """
    def default(self, obj):
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# =============================================================================
# 1. DOCUMENT LOADER
# =============================================================================

class DocumentLoader:
    """Reads PDF and text files from a folder and extracts their text content."""

    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)

    def load_documents(self) -> Dict[str, str]:
        """
        Load all PDF and text files from the folder.
        Returns: {filename: text_content}
        """
        if not self.folder_path.exists():
            print(f"  Creating folder: {self.folder_path}")
            self.folder_path.mkdir(parents=True, exist_ok=True)

        documents = {}

        supported_extensions = {'.pdf', '.txt'}
        all_files = [f for f in self.folder_path.iterdir()
                     if f.suffix.lower() in supported_extensions]

        if not all_files:
            return documents

        for file_path in sorted(all_files):
            print(f"  Loading: {file_path.name}")
            if file_path.suffix.lower() == '.pdf':
                content = self._read_pdf(file_path)
            else:
                content = self._read_text(file_path)

            if content.strip():
                documents[file_path.name] = content
            else:
                print(f"    Warning: {file_path.name} appears empty, skipping.")

        print(f"\n  Loaded {len(documents)} document(s) successfully.")
        return documents

    def _read_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using pdfplumber with PyPDF2 as fallback."""
        text = ""

        # Primary: pdfplumber (handles most modern PDFs well)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return text.strip()
        except Exception as e:
            print(f"    pdfplumber failed for {pdf_path.name}: {e}")

        # Fallback: PyPDF2
        try:
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"    PyPDF2 fallback also failed for {pdf_path.name}: {e}")

        return text.strip()

    def _read_text(self, txt_path: Path) -> str:
        """Read plain text file with UTF-8 encoding."""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except UnicodeDecodeError:
            # Fallback for files with different encoding
            with open(txt_path, 'r', encoding='latin-1') as f:
                return f.read().strip()
        except Exception as e:
            print(f"    Error reading {txt_path.name}: {e}")
            return ""


# =============================================================================
# 2. DOCUMENT CLASSIFIER
# =============================================================================

class DocumentClassifier:
    """
    Classifies documents into: Invoice, Resume, Utility Bill, Tax, Other, Unclassifiable.
    Uses a hybrid approach: keyword/pattern matching + semantic embedding similarity.
    """

    def __init__(self):
        print("  Loading classification model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Each category: keywords (weighted 2x) + regex patterns (weighted 3x)
        self.category_patterns = {
            'Invoice': {
                'keywords': [
                    'invoice', 'bill to', 'payment due', 'total amount',
                    'invoice number', 'subtotal', 'tax', 'amount due',
                    'purchase order', 'billing address', 'payment terms',
                    'line item', 'qty', 'unit price', 'vat'
                ],
                'patterns': [
                    r'inv[-\s]?\d+',                        # INV-1234 or INV 1234
                    r'invoice\s*[#no\.]*\s*:?\s*[\w\-]+',  # Invoice #: ABC-001
                    r'total\s*:?\s*\$?\s*\d+[.,]\d{2}',    # Total: $1,250.00
                    r'bill\s*to\s*:',                        # Bill To:
                    r'payment\s*due\s*:',                    # Payment Due:
                ]
            },
            'Resume': {
                'keywords': [
                    'experience', 'education', 'skills', 'work history',
                    'curriculum vitae', 'professional summary', 'objective',
                    'references', 'certifications', 'employment', 'projects',
                    'achievements', 'languages', 'linkedin', 'github',
                    'bachelor', 'master', 'degree', 'university', 'gpa'
                ],
                'patterns': [
                    r'\b\d{4}\s*[-–]\s*(?:\d{4}|present)',  # 2020 - Present
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',  # email
                    r'\+?\d[\d\s\-().]{8,}\d',              # phone number
                    r'\d+\+?\s*years?\s*(?:of\s*)?experience',
                    r'(?:bachelor|master|b\.?s\.?|m\.?s\.?|b\.?e\.?)',
                ]
            },
            'Utility Bill': {
                'keywords': [
                    'electricity', 'electric bill', 'water bill', 'gas bill',
                    'utility', 'account number', 'usage', 'kwh', 'kilowatt',
                    'meter reading', 'service address', 'billing period',
                    'current charges', 'previous balance', 'cubic feet',
                    'therms', 'consumption', 'rate per kwh'
                ],
                'patterns': [
                    r'account\s*[#no\.]*\s*:?\s*\d+',      # Account #: 12345
                    r'\d+[\.,]?\d*\s*kwh',                  # 845 kWh
                    r'meter\s*(?:number|#|no)',              # Meter Number
                    r'service\s*(?:address|period)',         # Service Address
                    r'billing\s*period\s*:',                 # Billing Period:
                ]
            },
            'Tax': {
                'keywords': [
                    'tax return', 'income tax', 'tax form', 'irs', 'w-2', 'w2',
                    '1099', '1040', 'adjusted gross income', 'taxable income',
                    'withholding', 'deductions', 'exemptions', 'filing status',
                    'federal tax', 'state tax', 'tax year', 'refund', 'owed',
                    'schedule', 'ein', 'tin', 'social security'
                ],
                'patterns': [
                    r'form\s*(?:1040|1099|w-?2|w-?4)',      # Form 1040 / W-2
                    r'tax\s*year\s*:?\s*\d{4}',             # Tax Year: 2024
                    r'(?:federal|state)\s*tax',              # Federal Tax
                    r'adjusted\s*gross\s*income',            # AGI
                    r'social\s*security\s*(?:number|no|#)',  # SSN
                ]
            }
        }

        # Build prototype embeddings for semantic similarity
        print("  Building category prototype embeddings...")
        self.category_prototypes = {}
        for category, data in self.category_patterns.items():
            prototype_text = " ".join(data['keywords'] * 2)
            self.category_prototypes[category] = self.model.encode([prototype_text])[0]

    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify a document and return (category, confidence_score).
        Uses weighted combination of keyword matching and semantic similarity.
        """
        if not text or len(text.strip()) < 30:
            return "Unclassifiable", 0.0

        text_lower = text.lower()
        scores = {cat: 0.0 for cat in self.category_patterns}

        # --- Method 1: Keyword + Regex scoring ---
        for category, data in self.category_patterns.items():
            kw_score = sum(2 for kw in data['keywords'] if kw in text_lower)
            pat_score = sum(3 for pat in data['patterns']
                           if re.search(pat, text_lower, re.IGNORECASE))
            # Normalize by max possible score
            max_possible = (len(data['keywords']) * 2) + (len(data['patterns']) * 3)
            scores[category] = (kw_score + pat_score) / max_possible

        # --- Method 2: Semantic embedding similarity ---
        doc_embedding = self.model.encode([text[:2000]])[0]  # Limit to 2000 chars for speed
        for category, prototype in self.category_prototypes.items():
            similarity = float(cosine_similarity([doc_embedding], [prototype])[0][0])
            # Weighted blend: 50% keyword, 50% semantic
            scores[category] = (scores[category] * 0.50) + (similarity * 0.50)

        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        # Confidence threshold: if too low, mark as unclassifiable
        if best_score < 0.15:
            return "Unclassifiable", round(best_score, 4)

        # If winning margin is tiny (< 0.02), call it "Other"
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) < 0.02:
            return "Other", round(best_score, 4)

        return best_category, round(best_score, 4)


# =============================================================================
# 3. INFORMATION EXTRACTOR
# =============================================================================

class InformationExtractor:
    """
    Extracts structured fields from documents based on their classified type.
    Uses regex patterns ordered from most-specific to least-specific.
    """

    # ------------------------------------------------------------------
    # Invoice Extraction
    # ------------------------------------------------------------------

    def extract_invoice(self, text: str) -> Dict:
        """Extract: invoice_number, date, company, total_amount"""

        # --- Invoice Number ---
        # Order matters: most specific first to avoid grabbing company names
        invoice_number = None
        invoice_patterns = [
            r'invoice\s*(?:#|no\.?|number)\s*:?\s*([A-Z0-9][\w\-/]{2,20})',
            r'inv[-\s]?([A-Z0-9][\w\-]{2,15})',
            r'order\s*(?:#|no\.?|number)\s*:?\s*([A-Z0-9][\w\-]{2,15})',
        ]
        for pattern in invoice_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                # Reject if it's clearly a company name (too long, has spaces, no digits)
                if len(candidate) <= 20 and re.search(r'\d', candidate):
                    invoice_number = candidate
                    break

        # --- Date ---
        date = None
        date_patterns = [
            # ISO: 2025-01-15
            r'(?:invoice\s*)?date\s*:?\s*(\d{4}[-/]\d{2}[-/]\d{2})',
            # US: 01/15/2025 or 01-15-2025
            r'(?:invoice\s*)?date\s*:?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
            # Written: January 15, 2025
            r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+'
             r'\d{1,2},?\s+\d{4})',
            # Any date-like without label
            r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})\b',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date = match.group(1).strip()
                break

        # --- Company ---
        company = None
        company_patterns = [
            r'(?:from|vendor|seller|issued\s*by)\s*:?\s*\n?\s*([A-Z][A-Za-z0-9\s&.,\-]{2,50})',
            r'(?:company|business)\s*(?:name)?\s*:?\s*([A-Z][A-Za-z0-9\s&.,\-]{2,50})',
            r'^([A-Z][A-Za-z0-9\s&.,]{2,40}(?:Ltd|LLC|Inc|Corp|Co|Limited)\.?)',
        ]
        for pattern in company_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                candidate = match.group(1).strip().rstrip(',.')
                # Sanity check: shouldn't be too short or a common word
                if len(candidate) > 3 and candidate.lower() not in ('invoice', 'date', 'total'):
                    company = candidate
                    break

        # --- Total Amount ---
        total_amount = None
        amount_patterns = [
            r'(?:grand\s*)?total\s*(?:amount\s*)?(?:due\s*)?:?\s*\$?\s*([\d,]+\.\d{2})',
            r'amount\s*(?:due|payable)\s*:?\s*\$?\s*([\d,]+\.\d{2})',
            r'balance\s*due\s*:?\s*\$?\s*([\d,]+\.\d{2})',
        ]
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the last match (usually the final total after line items)
                amount_str = matches[-1].replace(',', '')
                try:
                    total_amount = float(amount_str)
                except ValueError:
                    pass
                break

        return {
            'invoice_number': invoice_number,
            'date': date,
            'company': company,
            'total_amount': total_amount
        }

    # ------------------------------------------------------------------
    # Resume Extraction
    # ------------------------------------------------------------------

    def extract_resume(self, text: str) -> Dict:
        """Extract: name, email, phone, experience_years"""

        # --- Email (find first) ---
        email = None
        email_match = re.search(
            r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b', text)
        if email_match:
            email = email_match.group(0).lower()

        # --- Phone ---
        phone = None
        phone_patterns = [
            r'\+\d{1,3}[\s\-.]?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}',  # +92 300 1234567
            r'\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}',                     # (123) 456-7890
            r'\+\d{10,13}',                                                  # +923001234567
        ]
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                phone = match.group(0).strip()
                break

        # --- Name ---
        # Strategy: scan the first 15 lines for a short, title-case or ALL-CAPS line
        # that doesn't look like a section header or contact info
        name = None
        skip_words = {
            'resume', 'cv', 'curriculum', 'vitae', 'summary', 'objective',
            'profile', 'experience', 'education', 'skills', 'contact',
            'address', 'email', 'phone', 'linkedin', 'github', 'portfolio'
        }
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in lines[:15]:
            # Skip lines with '@', digits (phone/date), or URL-like content
            if '@' in line or re.search(r'\d{4}', line) or 'http' in line.lower():
                continue
            # Skip lines that are clearly section headers or too long
            if len(line) > 60 or len(line.split()) > 5:
                continue
            # Must have at least 2 words (first + last name)
            words = line.split()
            if len(words) < 2:
                continue
            # Skip if any word is a known section header
            if any(w.lower() in skip_words for w in words):
                continue
            # Accept ALL-CAPS or Title Case names
            if line.isupper() or line.istitle() or all(w[0].isupper() for w in words if w):
                name = line
                break

        # --- Experience Years ---
        experience_years = None
        exp_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)[\s\w]{0,10}experience',
            r'experience\s*(?:of\s*)?(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:professional|work|industry)',
        ]
        for pattern in exp_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                experience_years = int(match.group(1))
                break

        # Fallback: infer from earliest employment year in document
        if experience_years is None:
            year_matches = re.findall(r'\b(20\d{2}|19\d{2})\b', text)
            if len(year_matches) >= 2:
                years = sorted([int(y) for y in year_matches])
                current_year = datetime.now().year
                if years[0] < current_year:
                    inferred = current_year - years[0]
                    if 1 <= inferred <= 50:  # Sanity range
                        experience_years = inferred

        return {
            'name': name,
            'email': email,
            'phone': phone,
            'experience_years': experience_years
        }

    # ------------------------------------------------------------------
    # Utility Bill Extraction
    # ------------------------------------------------------------------

    def extract_utility_bill(self, text: str) -> Dict:
        """Extract: account_number, date, usage_kwh, amount_due"""

        # --- Account Number ---
        account_number = None
        account_patterns = [
            r'account\s*(?:#|no\.?|number)\s*:?\s*(\d[\d\s\-]{4,20})',
            r'customer\s*(?:#|no\.?|id|number)\s*:?\s*(\d[\d\s\-]{4,20})',
            r'meter\s*(?:#|no\.?|number)\s*:?\s*(\d[\d\s\-]{4,20})',
        ]
        for pattern in account_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                account_number = match.group(1).strip()
                break

        # --- Date (statement/bill date) ---
        date = None
        date_patterns = [
            r'(?:statement|bill|invoice|service)\s*date\s*:?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
            r'(?:billing|due)\s*date\s*:?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
            r'date\s*:?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
            r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date = match.group(1).strip()
                break

        # --- Usage in kWh ---
        usage_kwh = None
        usage_patterns = [
            r'(?:total\s*)?usage\s*:?\s*([\d,]+(?:\.\d+)?)\s*kwh',
            r'(?:energy\s*)?consumption\s*:?\s*([\d,]+(?:\.\d+)?)\s*kwh',
            r'([\d,]+(?:\.\d+)?)\s*kwh',  # bare "845 kWh"
        ]
        for pattern in usage_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    usage_kwh = float(match.group(1).replace(',', ''))
                except ValueError:
                    pass
                break

        # --- Amount Due ---
        amount_due = None
        amount_patterns = [
            r'(?:total\s*)?amount\s*due\s*:?\s*\$?\s*([\d,]+\.\d{2})',
            r'(?:total\s*)?balance\s*due\s*:?\s*\$?\s*([\d,]+\.\d{2})',
            r'(?:please\s*pay\s*)?total\s*:?\s*\$?\s*([\d,]+\.\d{2})',
            r'payment\s*due\s*:?\s*\$?\s*([\d,]+\.\d{2})',
        ]
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    amount_due = float(match.group(1).replace(',', ''))
                except ValueError:
                    pass
                break

        return {
            'account_number': account_number,
            'date': date,
            'usage_kwh': usage_kwh,
            'amount_due': amount_due
        }

    # ------------------------------------------------------------------
    # Tax Document Extraction
    # ------------------------------------------------------------------

    def extract_tax(self, text: str) -> Dict:
        """Extract basic fields from tax documents"""

        # Tax year
        tax_year = None
        year_match = re.search(r'tax\s*year\s*:?\s*(\d{4})', text, re.IGNORECASE)
        if not year_match:
            year_match = re.search(r'(?:for|year)\s*(\d{4})', text, re.IGNORECASE)
        if year_match:
            tax_year = year_match.group(1)

        # Total tax / amount
        amount = None
        amount_patterns = [
            r'total\s*tax\s*:?\s*\$?\s*([\d,]+\.\d{2})',
            r'amount\s*(?:owed|due|refund)\s*:?\s*\$?\s*([\d,]+\.\d{2})',
            r'tax\s*(?:due|owed)\s*:?\s*\$?\s*([\d,]+\.\d{2})',
        ]
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    amount = float(match.group(1).replace(',', ''))
                except ValueError:
                    pass
                break

        return {
            'tax_year': tax_year,
            'total_tax': amount
        }

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    def extract(self, text: str, category: str) -> Dict:
        """Route to the correct extractor based on document category."""
        extractors = {
            'Invoice':      self.extract_invoice,
            'Resume':       self.extract_resume,
            'Utility Bill': self.extract_utility_bill,
            'Tax':          self.extract_tax,
        }
        extractor_fn = extractors.get(category)
        if extractor_fn:
            return extractor_fn(text)
        return {}  # Other / Unclassifiable: no extraction required


# =============================================================================
# 4. SEMANTIC SEARCH
# =============================================================================

class SemanticSearch:
    """
    Local semantic search engine using FAISS + SentenceTransformers.
    Documents are chunked for better retrieval granularity.
    """

    CHUNK_SIZE = 400      # characters per chunk
    CHUNK_OVERLAP = 80    # overlap between consecutive chunks

    def __init__(self):
        # Reuse the same model already loaded (could inject, but keeping standalone)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[str] = []
        self.chunk_sources: List[str] = []  # filename for each chunk

    def build_index(self, documents: Dict[str, str]) -> None:
        """Chunk all documents, encode them, and build a FAISS L2 index."""
        self.chunks = []
        self.chunk_sources = []

        for filename, content in documents.items():
            for chunk in self._chunk_text(content):
                self.chunks.append(chunk)
                self.chunk_sources.append(filename)

        if not self.chunks:
            print("  No content to index.")
            return

        print(f"  Encoding {len(self.chunks)} chunks...")
        embeddings = self.model.encode(
            self.chunks, show_progress_bar=True, batch_size=32
        ).astype('float32')

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print(f"  FAISS index built: {self.index.ntotal} vectors, dim={dimension}")

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping character-level chunks."""
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + self.CHUNK_SIZE, text_len)
            chunks.append(text[start:end].strip())
            if end == text_len:
                break
            start += self.CHUNK_SIZE - self.CHUNK_OVERLAP
        return [c for c in chunks if len(c) > 20]  # Skip tiny remnants

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search documents by semantic meaning.
        Returns top_k unique results with relevance scores and excerpts.
        """
        if self.index is None or self.index.ntotal == 0:
            print("  Search index is empty. Did build_index() run successfully?")
            return []

        query_vec = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vec, min(top_k * 3, self.index.ntotal))

        results = []
        seen_files = set()

        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
            filename = self.chunk_sources[idx]
            if filename in seen_files:
                continue
            seen_files.add(filename)
            relevance = round(float(1.0 / (1.0 + dist)), 4)
            excerpt = self.chunks[idx]
            if len(excerpt) > 300:
                excerpt = excerpt[:300] + "..."
            results.append({
                'filename': filename,
                'relevance_score': relevance,
                'excerpt': excerpt
            })
            if len(results) >= top_k:
                break

        return results


# =============================================================================
# 5. BONUS: LOCAL QUESTION ANSWERING (optional)
# =============================================================================

class LocalQuestionAnswerer:
    """
    Optional bonus: uses a local open-source LLM for question answering.
    Uses microsoft/phi-2 by default (small, CPU-runnable).
    """

    def __init__(self, model_name: str = "microsoft/phi-2"):
        if not LOCAL_LLM_AVAILABLE:
            print("  Bonus QA disabled: install transformers and torch to enable.")
            self.model = None
            self.tokenizer = None
            return

        print(f"\n  Loading local LLM '{model_name}' (first run downloads the model)...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )
            self.model.eval()
            print("  Local LLM loaded successfully!")
        except Exception as e:
            print(f"  Could not load local LLM: {e}")
            self.model = None
            self.tokenizer = None

    def answer(self, question: str, context: str) -> str:
        if self.model is None:
            return ("Local LLM not available. "
                    "Install 'transformers' and 'torch' and restart.")

        prompt = (
            f"Read the context below and answer the question concisely.\n\n"
            f"Context:\n{context[:1500]}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # Extract only the answer portion
            answer_part = full_output.split("Answer:")[-1].strip()
            return answer_part if answer_part else "Could not generate an answer."
        except Exception as e:
            return f"Error during answer generation: {e}"


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 65)
    print("  LOCAL AI DOCUMENT PROCESSING SYSTEM")
    print("  AI Engineer Technical Assessment")
    print("=" * 65)

    INPUT_FOLDER = "./documents"
    OUTPUT_FILE = "output.json"

    # ------------------------------------------------------------------
    # STEP 1: Load documents
    # ------------------------------------------------------------------
    print("\n📁  STEP 1: Ingesting documents from folder...")
    loader = DocumentLoader(INPUT_FOLDER)
    documents = loader.load_documents()

    if not documents:
        print(f"\n  ⚠️  No PDF or TXT files found in '{INPUT_FOLDER}'.")
        print("  Please add your documents there and re-run.")
        return

    # ------------------------------------------------------------------
    # STEP 2: Classify + Extract
    # ------------------------------------------------------------------
    print("\n🏷️   STEP 2: Classifying and extracting structured data...")
    classifier = DocumentClassifier()
    extractor = InformationExtractor()

    output = {}

    for filename, content in documents.items():
        print(f"\n  ▶ {filename}")

        category, confidence = classifier.classify(content)
        print(f"    Class      : {category}  (confidence: {confidence:.4f})")

        extracted = extractor.extract(content, category)
        if extracted:
            for k, v in extracted.items():
                print(f"    {k:<25}: {v}")

        # Build output record — matches the required JSON schema
        record = {
            "class": category,
            "classification_confidence": confidence,
        }
        record.update(extracted)
        output[filename] = record

    # ------------------------------------------------------------------
    # STEP 3: Save output.json
    # (use NumpyEncoder to handle float32/int64 from numpy)
    # ------------------------------------------------------------------
    print(f"\n💾  STEP 3: Saving results to '{OUTPUT_FILE}'...")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"  ✅  Saved {len(output)} document record(s) to {OUTPUT_FILE}")
    except Exception as e:
        print(f"  ❌  Failed to save JSON: {e}")
        return

    # ------------------------------------------------------------------
    # STEP 4: Build semantic search index
    # ------------------------------------------------------------------
    print("\n🔍  STEP 4: Building semantic search index...")
    searcher = SemanticSearch()
    searcher.build_index(documents)

    # ------------------------------------------------------------------
    # STEP 5: Interactive CLI
    # ------------------------------------------------------------------
    print("\n💬  STEP 5: Interactive Semantic Search")
    print("─" * 65)
    print("  Commands:")
    print("    <query>   — search documents by meaning")
    print("    qa        — enter question-answering mode (bonus)")
    print("    quit      — exit")
    print("─" * 65)

    # Only load QA model if user explicitly requests it (saves startup time)
    qa_system = None

    while True:
        try:
            raw = input("\n🔎  Search: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting.")
            break

        if not raw:
            continue

        if raw.lower() in ('quit', 'exit', 'q'):
            break

        elif raw.lower() == 'qa':
            # Lazy-load QA model on first use
            if qa_system is None:
                if LOCAL_LLM_AVAILABLE:
                    qa_system = LocalQuestionAnswerer()
                else:
                    print("  ⚠️  QA feature requires 'transformers' and 'torch'. "
                          "Please install them.")
                    continue

            if qa_system.model is None:
                print("  ⚠️  QA model failed to load. See error above.")
                continue

            try:
                question = input("❓  Question: ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not question:
                continue

            context_results = searcher.search(question, top_k=3)
            if context_results:
                context = "\n\n".join(r['excerpt'] for r in context_results)
                print("\n  ⏳ Generating answer (may take a moment on CPU)...")
                answer = qa_system.answer(question, context)
                print(f"\n  ✅  Answer: {answer}")
                sources = [r['filename'] for r in context_results]
                print(f"  📚  Sources: {sources}")
            else:
                print("  No relevant documents found for that question.")

        else:
            # Normal semantic search
            results = searcher.search(raw, top_k=5)
            if results:
                print(f"\n  📄  Top {len(results)} result(s) for: \"{raw}\"\n")
                for i, r in enumerate(results, 1):
                    print(f"  {i}. {r['filename']}  "
                          f"(relevance: {r['relevance_score']:.4f})")
                    print(f"     {r['excerpt']}\n")
            else:
                print("  No relevant documents found.")

    print(f"\n✅  Done. All results saved to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()