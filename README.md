# Multimodal AI Document Analyzer

A premium Streamlit dashboard for analyzing PDFs, scans, images, invoices, resumes, handwritten notes, charts, and research papers.

## Features

- Drag-and-drop uploads for PDF, PNG, JPG, JPEG, TIFF, BMP, and WEBP
- PDF text extraction with PyMuPDF and pdfplumber
- OCR fallback for scanned PDFs and images through Tesseract
- Document previews, analytics cards, document memory, semantic-style search, and chat history
- AI summaries, citations, key insights, action items, timelines, emotional tone, resume matching, research simplification, and multimodal image-plus-text explanations
- Side-by-side document comparison
- JSON/TinyDB local persistence in `storage/`
- Groq or OpenAI support, with deterministic offline fallbacks when no API key is configured

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

For OCR, install Tesseract separately and make sure it is available on PATH.

Optional API keys:

```powershell
$env:GROQ_API_KEY="your_groq_key"
$env:OPENAI_API_KEY="your_openai_key"
```

Groq is preferred when both keys are present.
