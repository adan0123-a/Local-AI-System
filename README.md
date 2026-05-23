# Local AI Document Processing System
 

A fully offline document intelligence pipeline that ingests PDFs and text files, classifies them, extracts structured data, and enables semantic search — using only open-source libraries.

---

## Features

| Step | What it does |
|------|-------------|
| **Ingest** | Reads all `.pdf` and `.txt` files from `./documents/` |
| **Classify** | Labels each document as Invoice / Resume / Utility Bill / Tax / Other / Unclassifiable |
| **Extract** | Pulls structured fields per document type into `output.json` |
| **Search** | FAISS-powered semantic search — find documents by meaning, not just keywords |
| **QA (bonus)** | Local LLM (Phi-2) answers natural-language questions over retrieved context |

All processing runs **100% locally** — no internet, no API keys required after initial model download.

---

## Installation

### 1. Clone / unzip the project
```bash
cd local-ai-document-system
```

### 2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** If you want the optional bonus QA feature, also install:
> ```bash
> pip install torch transformers
> ```
> The base system works without these.

---

## Running the Program

### 1. Add your documents
Place all `.pdf` or `.txt` files into the `documents/` folder:
```
documents/
  invoice_1.pdf
  resume_john.pdf
  utility_bill_march.pdf
  ...
```

### 2. Run the pipeline
```bash
python app.py
```

### 3. What happens
1. Documents are loaded and text is extracted
2. Each document is classified and structured fields are extracted
3. Results are saved to `output.json`
4. A semantic search CLI starts — type any query and press Enter

### 4. Using the CLI
```
🔎 Search: invoices with payments due in January
🔎 Search: resumes with Python experience
🔎 Search: qa           ← enters bonus QA mode
🔎 Search: quit         ← exits
```

---

## Output Format (`output.json`)

```json
{
  "invoice_1.pdf": {
    "class": "Invoice",
    "classification_confidence": 0.4312,
    "invoice_number": "INV-1234",
    "date": "2025-01-01",
    "company": "ACME Ltd.",
    "total_amount": 350.50
  },
  "resume_john.pdf": {
    "class": "Resume",
    "classification_confidence": 0.3891,
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "+1 123-456-7890",
    "experience_years": 5
  },
  "electric_bill.pdf": {
    "class": "Utility Bill",
    "classification_confidence": 0.4921,
    "account_number": "98765432",
    "date": "03/15/2025",
    "usage_kwh": 845.0,
    "amount_due": 150.00
  }
}
```

---

## Libraries and Methods Used

### Document Ingestion
| Library | Purpose |
|---------|---------|
| `pdfplumber` | Primary PDF text extraction — handles complex layouts |
| `PyPDF2` | Fallback PDF reader for fonts pdfplumber can't parse |

### Classification
| Component | Description |
|-----------|-------------|
| `sentence-transformers` (`all-MiniLM-L6-v2`) | Generates 384-dim semantic embeddings for each document |
| Keyword scoring | Weighted count of domain-specific keywords per category |
| Regex pattern scoring | Structural patterns (e.g., `INV-\d+`, `\d+ kWh`) per category |
| Hybrid blend | 50% semantic similarity + 50% keyword/regex score |

**Categories:** Invoice, Resume, Utility Bill, Tax, Other, Unclassifiable

### Information Extraction
Pure regex-based extraction with patterns ordered from most-specific to least-specific:
- Invoices: invoice number, date, company name, total amount
- Resumes: name, email, phone, years of experience (with year-inference fallback)
- Utility Bills: account number, billing date, kWh usage, amount due
- Tax Documents: tax year, total tax amount

### Semantic Search
| Component | Description |
|-----------|-------------|
| `sentence-transformers` | Encodes document chunks and queries into dense vectors |
| `faiss` (`IndexFlatL2`) | Exact nearest-neighbour search over all chunk embeddings |
| Chunking | 400-char chunks with 80-char overlap for fine-grained retrieval |

### Bonus QA (optional)
| Component | Description |
|-----------|-------------|
| `transformers` | Loads `microsoft/phi-2` — a small LLM that runs on CPU |
| `torch` | PyTorch inference backend |
| RAG pattern | Top-3 retrieved chunks → context → LLM answer |

---

## Requirements (`requirements.txt`)

```
pdfplumber>=0.9.0
PyPDF2>=3.0.0
sentence-transformers>=2.2.0
scikit-learn>=1.0.0
faiss-cpu>=1.7.0
numpy>=1.23.0
```

Optional (bonus QA):
```
torch>=2.0.0
transformers>=4.30.0
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `float32 is not JSON serializable` | Fixed via `NumpyEncoder` — should not occur |
| `invoice1.pdf appears empty` | File has no extractable text (scanned image PDF) |
| `FontBBox warnings` | Harmless font metadata issue in PyPDF2; suppressed automatically |
| Slow startup | Model downloads on first run (~90 MB); cached locally afterward |
| QA too slow | Phi-2 runs on CPU; expect 30–90s per answer. Use GPU for speed |

---

## Project Structure

```
local-ai-document-system/
├── app.py              ← Main pipeline (all components in one file)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
├── output.json         ← Generated after running app.py
└── documents/          ← Place your PDF/TXT files here
    ├── invoice_1.pdf
    ├── resume_1.pdf
    └── ...
```