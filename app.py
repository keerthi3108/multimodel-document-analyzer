from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import shutil
import textwrap
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import pandas as pd
import pdfplumber
import plotly.express as px
import plotly.graph_objects as go
import pytesseract
import streamlit as st
from PIL import Image
from tinydb import Query, TinyDB

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional in constrained installs
    load_dotenv = None

try:
    from langsmith import traceable
except Exception:  # pragma: no cover - LangSmith is optional at runtime
    def traceable(*_args: Any, **_kwargs: Any):
        def decorator(func):
            return func
        return decorator


APP_TITLE = "NexusDoc AI"
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
REPORT_DIR = STORAGE_DIR / "reports"
DB_PATH = STORAGE_DIR / "nexusdoc.json"
SUPPORTED_TYPES = ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"]
IMAGE_TYPES = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


@dataclass
class ProcessedDocument:
    doc_id: str
    name: str
    file_type: str
    path: str
    uploaded_at: str
    text: str
    pages: list[dict[str, Any]]
    metadata: dict[str, Any]
    analysis: dict[str, Any]


def bootstrap() -> TinyDB:
    if load_dotenv:
        load_dotenv()
    os.environ.setdefault("LANGSMITH_PROJECT", "nexusdoc-ai")
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")
    STORAGE_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    return TinyDB(DB_PATH, indent=2)


db = bootstrap()
documents_table = db.table("documents")
chat_table = db.table("chat")
report_table = db.table("reports")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111f;
            --panel: rgba(12, 24, 43, .84);
            --panel-2: rgba(18, 33, 56, .78);
            --line: rgba(148, 163, 184, .20);
            --text: #edf4ff;
            --muted: #9fb0ca;
            --accent: #35d0ba;
            --accent-2: #8fb7ff;
            --warn: #ffcb74;
        }

        .stApp {
            color: var(--text);
            background:
                radial-gradient(circle at 22% 18%, rgba(53,208,186,.18), transparent 28%),
                radial-gradient(circle at 78% 8%, rgba(143,183,255,.15), transparent 24%),
                linear-gradient(135deg, #050b14 0%, #081323 48%, #0d1425 100%);
        }

        .stApp, .stApp p, .stApp li, .stApp label, .stApp span,
        .stApp div, .stApp textarea, .stApp input {
            color: #edf4ff;
        }

        [data-testid="stSidebar"] {
            background: #050c18;
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #dbeafe !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #c8d6ee !important;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #f1f7ff !important;
        }

        [data-testid="stHeader"] {
            background: rgba(5, 12, 24, .82);
            backdrop-filter: blur(14px);
            border-bottom: 1px solid rgba(148, 163, 184, .14);
        }

        [data-testid="stFileUploaderDropzone"] {
            background: rgba(12, 24, 43, .96);
            border: 1px dashed rgba(143, 183, 255, .34);
            border-radius: 8px;
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: #dbeafe !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            background: rgba(53, 208, 186, .14);
            border: 1px solid rgba(53, 208, 186, .34);
            color: #eafefa;
        }

        h1, h2, h3 {
            letter-spacing: 0;
            color: #f4f8ff;
        }

        .stMarkdown, .stMarkdown p, .stCaptionContainer, .stText, .stTextArea label {
            color: #dbe7fb !important;
        }

        textarea, .stTextArea textarea {
            background: #071426 !important;
            color: #f4f8ff !important;
            border: 1px solid rgba(143, 183, 255, .34) !important;
            caret-color: #35d0ba !important;
        }

        textarea::placeholder, input::placeholder {
            color: #9fb0ca !important;
            opacity: 1 !important;
        }

        input, .stTextInput input {
            background: #071426 !important;
            color: #f4f8ff !important;
            border-color: rgba(143, 183, 255, .34) !important;
        }

        div[data-testid="stAlert"] p, div[data-testid="stAlert"] span {
            color: #eafefa !important;
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: #f4f8ff !important;
        }

        [data-baseweb="select"] *, [data-baseweb="radio"] *, [data-baseweb="checkbox"] * {
            color: #edf4ff !important;
        }

        [data-baseweb="select"],
        [data-baseweb="select"] > div {
            background: #071426 !important;
            border-color: rgba(143, 183, 255, .34) !important;
        }

        [data-baseweb="tab-highlight"] {
            background-color: #35d0ba !important;
        }

        [data-testid="stSegmentedControl"] {
            background: transparent !important;
        }

        [data-testid="stSegmentedControl"] label,
        [data-testid="stSegmentedControl"] label p {
            color: #dbe7fb !important;
        }

        [data-testid="stSegmentedControl"] label {
            background: #071426 !important;
            border-color: rgba(143, 183, 255, .30) !important;
        }

        [data-testid="stSegmentedControl"] label[data-baseweb="radio"] {
            color: #dbe7fb !important;
        }

        [data-testid="stSegmentedControl"] label:has(input:checked),
        [data-testid="stSegmentedControl"] [aria-checked="true"] {
            background: rgba(53, 208, 186, .18) !important;
            border-color: rgba(53, 208, 186, .62) !important;
        }

        [data-testid="stSegmentedControl"] label:has(input:checked) p,
        [data-testid="stSegmentedControl"] [aria-checked="true"] p {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        div[role="radiogroup"] {
            background: transparent !important;
        }

        div[role="radiogroup"] label,
        div[role="radiogroup"] label > div,
        div[role="radiogroup"] label > span {
            background: #071426 !important;
            color: #dbeafe !important;
            border-color: rgba(143, 183, 255, .34) !important;
        }

        div[role="radiogroup"] label p,
        div[role="radiogroup"] label span,
        div[role="radiogroup"] label div {
            color: #dbeafe !important;
        }

        div[role="radiogroup"] label:has(input:checked),
        div[role="radiogroup"] label:has(input:checked) > div,
        div[role="radiogroup"] label:has(input:checked) > span {
            background: rgba(53, 208, 186, .20) !important;
            color: #ffffff !important;
            border-color: rgba(53, 208, 186, .70) !important;
        }

        div[role="radiogroup"] label:has(input:checked) p,
        div[role="radiogroup"] label:has(input:checked) span,
        div[role="radiogroup"] label:has(input:checked) div {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        .hero {
            padding: 24px 26px;
            border: 1px solid var(--line);
            background: linear-gradient(135deg, rgba(18,33,56,.92), rgba(7,17,31,.72));
            border-radius: 8px;
            box-shadow: 0 24px 70px rgba(0,0,0,.26);
            position: relative;
            overflow: hidden;
        }

        .hero:after {
            content: "";
            position: absolute;
            inset: auto -80px -130px auto;
            width: 300px;
            height: 300px;
            background: conic-gradient(from 140deg, rgba(53,208,186,.35), rgba(143,183,255,.24), transparent);
            filter: blur(42px);
            animation: drift 8s ease-in-out infinite alternate;
        }

        @keyframes drift {
            from { transform: translate3d(0, 0, 0) rotate(0deg); }
            to { transform: translate3d(-30px, -20px, 0) rotate(18deg); }
        }

        .hero-title {
            font-size: clamp(2rem, 5vw, 4.6rem);
            line-height: 1.02;
            font-weight: 780;
            margin: 0;
            max-width: 1000px;
        }

        .hero-copy {
            color: #c8d6ee;
            font-size: 1rem;
            max-width: 820px;
            margin-top: 14px;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 11px;
            border: 1px solid rgba(53,208,186,.34);
            color: #bffdf3;
            background: rgba(53,208,186,.10);
            border-radius: 999px;
            font-size: .82rem;
            margin-bottom: 14px;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 16px 0 8px;
        }

        .metric-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 15px 16px;
            min-height: 104px;
            transition: transform .16s ease, border-color .16s ease;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(53,208,186,.40);
        }

        .metric-label {
            color: #b8c8e4;
            font-size: .78rem;
            text-transform: uppercase;
            letter-spacing: .06em;
        }

        .metric-value {
            font-size: 1.85rem;
            font-weight: 780;
            margin-top: 6px;
        }

        .metric-note {
            color: #b8c8e4;
            font-size: .82rem;
            margin-top: 4px;
        }

        .glass-panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px;
        }

        .insight-chip {
            display: inline-flex;
            align-items: center;
            margin: 4px 6px 4px 0;
            padding: 6px 9px;
            border-radius: 999px;
            border: 1px solid rgba(143,183,255,.25);
            background: rgba(143,183,255,.10);
            color: #d8e5ff;
            font-size: .82rem;
        }

        .doc-row {
            border: 1px solid var(--line);
            background: var(--panel-2);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 9px;
        }

        .small-muted {
            color: #b8c8e4;
            font-size: .84rem;
        }

        .stButton > button, .stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid rgba(53,208,186,.34);
            background: rgba(53,208,186,.12);
            color: #eafefa;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 1px solid var(--line);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            color: #cbd7ea;
            background: rgba(18, 33, 56, .46);
            padding: 10px 16px;
        }

        @media (max-width: 900px) {
            .metric-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 560px) {
            .metric-grid {
                grid-template-columns: 1fr;
            }
            .hero {
                padding: 18px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", normalize_text(text))
    return [p.strip() for p in parts if len(p.strip()) > 20]


def keywords(text: str, limit: int = 12) -> list[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "will",
        "have", "has", "had", "into", "your", "their", "about", "which", "also", "can",
        "not", "but", "all", "our", "you", "its", "use", "using", "than", "then", "there",
        "these", "those", "they", "them", "been", "may", "per", "page", "pages",
    }
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
    counts = Counter(w for w in words if w not in stop)
    return [word for word, _ in counts.most_common(limit)]


def simple_summary(text: str, max_sentences: int = 5) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return (
            "No text layer was detected yet. If this is an image, chart, scan, or handwritten page, "
            "use visual analysis or re-run analysis after Tesseract is available."
        )
    key_terms = set(keywords(text, 18))
    scored: list[tuple[int, int, str]] = []
    for idx, sentence in enumerate(sentences):
        score = sum(1 for term in key_terms if term in sentence.lower())
        score += min(len(sentence) // 130, 2)
        scored.append((score, -idx, sentence))
    selected = sorted(scored, reverse=True)[:max_sentences]
    ordered = sorted(selected, key=lambda item: -item[1])
    return " ".join(item[2] for item in ordered)


def detect_doc_type(name: str, text: str, file_type: str) -> str:
    lower = f"{name} {text[:4000]}".lower()
    if "invoice" in lower or "amount due" in lower or "tax invoice" in lower:
        return "Invoice"
    if "curriculum vitae" in lower or "resume" in lower or "linkedin" in lower:
        return "Resume"
    if "abstract" in lower and ("references" in lower or "methodology" in lower):
        return "Research Paper"
    if "chart" in lower or "figure" in lower or file_type in {"png", "jpg", "jpeg", "webp", "tif", "tiff"}:
        return "Image / Chart"
    if "meeting" in lower or "action item" in lower:
        return "Notes"
    return "Document"


def emotional_tone(text: str) -> dict[str, Any]:
    positive = {"growth", "success", "improve", "strong", "benefit", "effective", "excellent", "opportunity"}
    negative = {"risk", "decline", "issue", "problem", "delay", "failure", "loss", "urgent", "concern"}
    analytical = {"data", "analysis", "method", "evidence", "result", "model", "research", "metric"}
    words = set(re.findall(r"[A-Za-z]+", text.lower()))
    scores = {
        "Optimistic": len(words & positive),
        "Concerned": len(words & negative),
        "Analytical": len(words & analytical),
    }
    top = max(scores, key=scores.get) if any(scores.values()) else "Neutral"
    confidence = min(92, 48 + max(scores.values()) * 8) if top != "Neutral" else 45
    return {"label": top, "confidence": confidence, "scores": scores}


def extract_action_items(text: str, limit: int = 8) -> list[str]:
    patterns = ["must", "should", "need to", "needs to", "action", "follow up", "deadline", "required", "recommend"]
    items = []
    for sentence in split_sentences(text):
        if any(pattern in sentence.lower() for pattern in patterns):
            items.append(sentence)
    return items[:limit]


def extract_timeline(text: str) -> list[dict[str, str]]:
    date_patterns = [
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b20\d{2}\b",
    ]
    events = []
    seen = set()
    for sentence in split_sentences(text):
        for pattern in date_patterns:
            match = re.search(pattern, sentence, flags=re.I)
            if match and match.group(0) not in seen:
                seen.add(match.group(0))
                events.append({"date": match.group(0), "event": sentence[:220]})
                break
        if len(events) >= 8:
            break
    return events


def generate_citations(pages: list[dict[str, Any]], query: str = "") -> list[dict[str, Any]]:
    terms = set(keywords(query, 8)) if query else set(keywords(" ".join(p["text"] for p in pages), 10))
    citations = []
    for page in pages:
        sentences = split_sentences(page["text"])
        for sentence in sentences:
            sentence_terms = set(re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", sentence.lower()))
            if not terms or sentence_terms & terms:
                citations.append({"page": page["page"], "quote": sentence[:260]})
                break
        if len(citations) >= 6:
            break
    return citations


def analyze_text(name: str, text: str, pages: list[dict[str, Any]], file_type: str) -> dict[str, Any]:
    words = re.findall(r"\b\w+\b", text)
    doc_type = detect_doc_type(name, text, file_type)
    terms = keywords(text)
    action_items = extract_action_items(text)
    timeline = extract_timeline(text)
    tone = emotional_tone(text)
    citations = generate_citations(pages)
    reading_minutes = max(1, round(len(words) / 220)) if words else 0
    return {
        "doc_type": doc_type,
        "summary": simple_summary(text),
        "keywords": terms,
        "action_items": action_items,
        "timeline": timeline,
        "tone": tone,
        "citations": citations,
        "word_count": len(words),
        "character_count": len(text),
        "reading_minutes": reading_minutes,
        "quality_score": min(98, max(18, round((len(text) / 60) + len(pages) * 7))),
    }


def has_readable_text(text: str) -> bool:
    if not text:
        return False
    if "OCR unavailable:" in text:
        return False
    words = re.findall(r"[A-Za-z0-9]{2,}", text)
    return len(words) >= 8


def image_to_data_url(image: Image.Image, max_side: int = 1600) -> str:
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def document_image_data_urls(path: str, file_type: str, limit: int = 3) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    urls: list[str] = []
    if file_type == "pdf":
        doc = fitz.open(file_path)
        try:
            for page_index in range(min(limit, doc.page_count)):
                page = doc[page_index]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                urls.append(image_to_data_url(image))
        finally:
            doc.close()
        return urls
    if file_type in IMAGE_TYPES:
        image = Image.open(file_path)
        return [image_to_data_url(image)]
    return []


@traceable(name="nexusdoc_multimodal_llm_call", run_type="llm", project_name=os.getenv("LANGSMITH_PROJECT", "nexusdoc-ai"))
def call_multimodal_llm(prompt: str, doc: dict[str, Any] | ProcessedDocument, temperature: float = 0.2) -> str:
    provider, online = provider_status()
    if not online:
        context = doc.text if isinstance(doc, ProcessedDocument) else doc.get("text", "")
        return offline_answer(prompt, context)

    if isinstance(doc, ProcessedDocument):
        path = doc.path
        file_type = doc.file_type
        context = doc.text
        name = doc.name
    else:
        path = doc.get("path", "")
        file_type = doc.get("file_type", "")
        context = doc.get("text", "")
        name = doc.get("name", "document")

    image_urls = document_image_data_urls(path, file_type)
    if not image_urls:
        return call_llm(prompt, context, temperature)

    visual_prompt = (
        f"Document: {name}\n"
        f"Extract and explain both visual and text evidence. Identify document type, visible text, "
        f"layout, chart/table signals, key facts, timeline clues, citations/page references when possible, "
        f"and practical insights. Existing OCR/text layer:\n{context[:6000]}\n\nTask: {prompt}"
    )
    try:
        if provider == "Groq":
            from groq import Groq

            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            content: list[dict[str, Any]] = [{"type": "text", "text": visual_prompt}]
            for image_url in image_urls[:5]:
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            response = client.chat.completions.create(
                model=os.getenv("GROQ_VISION_MODEL", GROQ_VISION_MODEL),
                messages=[
                    {
                        "role": "system",
                        "content": "You are a multimodal document intelligence analyst. Be concise, factual, and cite page/image numbers when available.",
                    },
                    {"role": "user", "content": content},
                ],
                temperature=temperature,
                max_completion_tokens=1400,
            )
            return response.choices[0].message.content or ""

        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        content = [{"type": "text", "text": visual_prompt}]
        for image_url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
            messages=[
                {
                    "role": "system",
                    "content": "You are a multimodal document intelligence analyst. Be concise, factual, and cite page/image numbers when available.",
                },
                {"role": "user", "content": content},
            ],
            temperature=temperature,
            max_tokens=1400,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        return f"{offline_answer(prompt, context)}\n\nVision fallback note: {exc}"


def ocr_image(image: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(image)
    except Exception as exc:
        return f"[OCR unavailable: {exc}]"


def extract_pdf(path: Path, use_ocr: bool) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {"page_count": 0, "extractor": "PyMuPDF + pdfplumber"}
    fitz_doc = fitz.open(path)
    metadata["page_count"] = fitz_doc.page_count
    metadata["pdf_metadata"] = dict(fitz_doc.metadata or {})

    plumber_text: dict[int, str] = {}
    try:
        with pdfplumber.open(path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                plumber_text[idx] = page.extract_text() or ""
    except Exception:
        plumber_text = {}

    for index, page in enumerate(fitz_doc, start=1):
        text = page.get_text("text") or plumber_text.get(index, "")
        used_ocr = False
        if use_ocr and len(normalize_text(text)) < 40:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            text = f"{text}\n{ocr_image(image)}"
            used_ocr = True
        pages.append({"page": index, "text": normalize_text(text), "used_ocr": used_ocr})
    fitz_doc.close()
    return "\n\n".join(page["text"] for page in pages), pages, metadata


def extract_image(path: Path, use_ocr: bool) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    image = Image.open(path)
    metadata = {
        "page_count": 1,
        "extractor": "Pillow + Tesseract OCR",
        "image_size": f"{image.width} x {image.height}",
        "mode": image.mode,
    }
    text = ocr_image(image) if use_ocr else ""
    pages = [{"page": 1, "text": normalize_text(text), "used_ocr": use_ocr}]
    return normalize_text(text), pages, metadata


@traceable(name="nexusdoc_process_upload", run_type="chain", project_name=os.getenv("LANGSMITH_PROJECT", "nexusdoc-ai"))
def process_upload(uploaded_file: Any, use_ocr: bool) -> ProcessedDocument:
    content = uploaded_file.getvalue()
    doc_id = file_hash(content)
    extension = uploaded_file.name.split(".")[-1].lower()
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", uploaded_file.name)
    destination = UPLOAD_DIR / f"{doc_id}_{safe_name}"
    destination.write_bytes(content)

    if extension == "pdf":
        text, pages, metadata = extract_pdf(destination, use_ocr)
    else:
        text, pages, metadata = extract_image(destination, use_ocr)

    analysis = analyze_text(uploaded_file.name, text, pages, extension)
    record = ProcessedDocument(
        doc_id=doc_id,
        name=uploaded_file.name,
        file_type=extension,
        path=str(destination),
        uploaded_at=datetime.now().isoformat(timespec="seconds"),
        text=text,
        pages=pages,
        metadata=metadata,
        analysis=analysis,
    )
    if not has_readable_text(text) and extension in IMAGE_TYPES | {"pdf"}:
        visual_summary = call_multimodal_llm(
            "Create a document intelligence brief from the visual content. Extract any visible words, explain charts/tables/layout, and list key insights.",
            record,
        )
        if visual_summary and not visual_summary.startswith("No text layer"):
            record.text = normalize_text(f"{text}\n\n[Visual analysis]\n{visual_summary}")
            record.pages = [{"page": 1, "text": record.text, "used_ocr": use_ocr, "used_vision": True}]
            record.analysis = analyze_text(uploaded_file.name, record.text, record.pages, extension)
            record.analysis["visual_summary"] = visual_summary
    upsert_document(record)
    return record


def reprocess_document(doc: dict[str, Any], use_ocr: bool = True) -> dict[str, Any]:
    path = Path(doc.get("path", ""))
    if not path.exists():
        return doc
    extension = doc.get("file_type", path.suffix.lstrip(".").lower())
    if extension == "pdf":
        text, pages, metadata = extract_pdf(path, use_ocr)
    else:
        text, pages, metadata = extract_image(path, use_ocr)

    analysis = analyze_text(doc.get("name", path.name), text, pages, extension)
    updated = dict(doc)
    updated.update({"text": text, "pages": pages, "metadata": metadata, "analysis": analysis})
    if not has_readable_text(text) and extension in IMAGE_TYPES | {"pdf"}:
        temp_record = ProcessedDocument(
            doc_id=updated["doc_id"],
            name=updated["name"],
            file_type=extension,
            path=str(path),
            uploaded_at=updated.get("uploaded_at", datetime.now().isoformat(timespec="seconds")),
            text=text,
            pages=pages,
            metadata=metadata,
            analysis=analysis,
        )
        visual_summary = call_multimodal_llm(
            "Re-analyze this document visually. Extract visible text, explain image/chart/table content, and summarize insights.",
            temp_record,
        )
        if visual_summary and not visual_summary.startswith("No text layer"):
            updated["text"] = normalize_text(f"{text}\n\n[Visual analysis]\n{visual_summary}")
            updated["pages"] = [{"page": 1, "text": updated["text"], "used_ocr": use_ocr, "used_vision": True}]
            updated["analysis"] = analyze_text(updated["name"], updated["text"], updated["pages"], extension)
            updated["analysis"]["visual_summary"] = visual_summary
    Document = Query()
    documents_table.upsert(updated, Document.doc_id == updated["doc_id"])
    return updated


def upsert_document(record: ProcessedDocument) -> None:
    Document = Query()
    documents_table.upsert(record.__dict__, Document.doc_id == record.doc_id)


def get_documents() -> list[dict[str, Any]]:
    docs = documents_table.all()
    return sorted(docs, key=lambda item: item.get("uploaded_at", ""), reverse=True)


def get_document(doc_id: str | None) -> dict[str, Any] | None:
    if not doc_id:
        return None
    Document = Query()
    return documents_table.get(Document.doc_id == doc_id)


def provider_status() -> tuple[str, bool]:
    if os.getenv("GROQ_API_KEY"):
        return "Groq", True
    if os.getenv("OPENAI_API_KEY"):
        return "OpenAI", True
    return "Offline intelligence", False


def langsmith_status() -> tuple[str, bool]:
    has_key = bool(os.getenv("LANGSMITH_API_KEY"))
    tracing_on = os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes", "on"}
    if has_key and tracing_on:
        return os.getenv("LANGSMITH_PROJECT", "nexusdoc-ai"), True
    if has_key:
        return "LangSmith key found, tracing off", False
    return "LangSmith not configured", False


@traceable(name="nexusdoc_text_llm_call", run_type="llm", project_name=os.getenv("LANGSMITH_PROJECT", "nexusdoc-ai"))
def call_llm(prompt: str, context: str, temperature: float = 0.2) -> str:
    provider, online = provider_status()
    if not online:
        return offline_answer(prompt, context)
    try:
        if provider == "Groq":
            from groq import Groq

            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
                messages=[
                    {"role": "system", "content": "You are a precise document intelligence analyst. Cite page numbers when provided."},
                    {"role": "user", "content": f"Context:\n{context[:18000]}\n\nTask:\n{prompt}"},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a precise document intelligence analyst. Cite page numbers when provided."},
                {"role": "user", "content": f"Context:\n{context[:18000]}\n\nTask:\n{prompt}"},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        return f"{offline_answer(prompt, context)}\n\nAI provider fallback note: {exc}"


def offline_answer(prompt: str, context: str) -> str:
    prompt_lower = prompt.lower()
    if "timeline" in prompt_lower:
        timeline = extract_timeline(context)
        if not timeline:
            return "I did not find explicit dates in the available text."
        return "\n".join(f"- {item['date']}: {item['event']}" for item in timeline)
    if "action" in prompt_lower:
        items = extract_action_items(context)
        return "\n".join(f"- {item}" for item in items) if items else "No explicit action items were detected."
    if "tone" in prompt_lower or "emotion" in prompt_lower:
        tone = emotional_tone(context)
        return f"Detected tone: {tone['label']} ({tone['confidence']}% confidence). Score profile: {tone['scores']}."
    if "keyword" in prompt_lower or "insight" in prompt_lower:
        return "Key concepts: " + ", ".join(keywords(context, 14))
    if "summar" in prompt_lower or "simpl" in prompt_lower:
        return simple_summary(context, 7)
    return simple_summary(context, 4)


def page_context(doc: dict[str, Any]) -> str:
    chunks = []
    for page in doc.get("pages", []):
        chunks.append(f"[Page {page.get('page')}] {page.get('text', '')}")
    return "\n".join(chunks)


def semantic_search(query: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_terms = set(keywords(query, 16)) | set(re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", query.lower()))
    results = []
    for doc in docs:
        for page in doc.get("pages", []):
            text = page.get("text", "")
            page_terms = set(re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower()))
            overlap = query_terms & page_terms
            if overlap:
                score = len(overlap) / max(1, len(query_terms))
                snippet = best_snippet(text, query_terms)
                results.append(
                    {
                        "document": doc["name"],
                        "doc_id": doc["doc_id"],
                        "page": page.get("page"),
                        "score": score,
                        "snippet": snippet,
                    }
                )
    return sorted(results, key=lambda item: item["score"], reverse=True)[:10]


def best_snippet(text: str, terms: set[str]) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return text[:260]
    scored = []
    for sentence in sentences:
        score = sum(1 for term in terms if term in sentence.lower())
        scored.append((score, sentence))
    return max(scored, key=lambda item: item[0])[1][:340]


def compare_documents(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_terms = set(left["analysis"].get("keywords", []))
    right_terms = set(right["analysis"].get("keywords", []))
    common = sorted(left_terms & right_terms)
    left_only = sorted(left_terms - right_terms)
    right_only = sorted(right_terms - left_terms)
    similarity = round((len(common) / max(1, len(left_terms | right_terms))) * 100)
    return {
        "similarity": similarity,
        "common": common,
        "left_only": left_only,
        "right_only": right_only,
        "left_summary": left["analysis"].get("summary", ""),
        "right_summary": right["analysis"].get("summary", ""),
    }


def resume_match(resume: str, job_description: str) -> dict[str, Any]:
    resume_terms = set(keywords(resume, 80))
    job_terms = set(keywords(job_description, 80))
    matched = sorted(resume_terms & job_terms)
    missing = sorted(job_terms - resume_terms)[:18]
    score = round((len(matched) / max(1, len(job_terms))) * 100)
    return {"score": score, "matched": matched[:18], "missing": missing}


def make_report(doc: dict[str, Any]) -> Path:
    analysis = doc.get("analysis", {})
    report = {
        "document": doc.get("name"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": analysis.get("summary"),
        "keywords": analysis.get("keywords"),
        "action_items": analysis.get("action_items"),
        "timeline": analysis.get("timeline"),
        "citations": analysis.get("citations"),
        "tone": analysis.get("tone"),
        "visual_summary": analysis.get("visual_summary"),
    }
    path = REPORT_DIR / f"{doc['doc_id']}_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_table.insert({"doc_id": doc["doc_id"], "path": str(path), "created_at": report["generated_at"]})
    return path


def image_preview(path: str, file_type: str) -> None:
    p = Path(path)
    if not p.exists():
        st.warning("Preview file is missing.")
        return
    if file_type == "pdf":
        try:
            doc = fitz.open(p)
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False)
            st.image(pix.tobytes("png"), use_container_width=True)
            doc.close()
        except Exception as exc:
            st.info(f"PDF preview unavailable: {exc}")
    else:
        st.image(str(p), use_container_width=True)


def metric_cards(docs: list[dict[str, Any]]) -> None:
    total_pages = sum(int(doc.get("metadata", {}).get("page_count", 0) or 0) for doc in docs)
    total_words = sum(int(doc.get("analysis", {}).get("word_count", 0) or 0) for doc in docs)
    doc_types = Counter(doc.get("analysis", {}).get("doc_type", "Document") for doc in docs)
    avg_quality = round(np.mean([doc.get("analysis", {}).get("quality_score", 0) for doc in docs]) if docs else 0)
    top_type = doc_types.most_common(1)[0][0] if doc_types else "None"
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card"><div class="metric-label">Documents</div><div class="metric-value">{len(docs)}</div><div class="metric-note">In active memory</div></div>
            <div class="metric-card"><div class="metric-label">Pages</div><div class="metric-value">{total_pages}</div><div class="metric-note">Across uploads</div></div>
            <div class="metric-card"><div class="metric-label">Words</div><div class="metric-value">{total_words:,}</div><div class="metric-note">Extracted and indexed</div></div>
            <div class="metric-card"><div class="metric-label">Signal</div><div class="metric-value">{avg_quality}%</div><div class="metric-note">Top type: {top_type}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(docs: list[dict[str, Any]]) -> tuple[bool, list[Any]]:
    st.sidebar.markdown("## Command Center")
    provider, online = provider_status()
    langsmith_project, tracing_enabled = langsmith_status()
    st.sidebar.success(f"{provider} ready" if online else "Offline mode active")
    st.sidebar.caption(f"LangSmith tracing: {'on' if tracing_enabled else 'off'}")
    if tracing_enabled:
        st.sidebar.caption(f"Project: {langsmith_project}")
    use_ocr = st.sidebar.toggle("OCR for scans and images", value=True)
    uploaded = st.sidebar.file_uploader(
        "Drop documents",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="PDFs, scans, screenshots, handwritten notes, invoices, resumes, charts, and papers.",
    )
    st.sidebar.divider()
    st.sidebar.markdown("### Memory")
    st.sidebar.caption(f"{len(docs)} document(s) stored in JSON memory.")
    if st.sidebar.button("Clear chat history", use_container_width=True):
        chat_table.truncate()
        st.rerun()
    if st.sidebar.button("Export memory index", use_container_width=True):
        index_path = STORAGE_DIR / "memory_index.json"
        index_path.write_text(json.dumps(docs, indent=2), encoding="utf-8")
        st.sidebar.caption(f"Saved {index_path.name}")
    return use_ocr, uploaded


def render_upload_results(uploaded: list[Any], use_ocr: bool) -> None:
    if not uploaded:
        return
    if "processed_upload_ids" not in st.session_state:
        st.session_state.processed_upload_ids = set()

    Document = Query()
    pending = []
    for item in uploaded:
        upload_id = file_hash(item.getvalue())
        already_in_memory = documents_table.contains(Document.doc_id == upload_id)
        already_processed = upload_id in st.session_state.processed_upload_ids
        if already_in_memory or already_processed:
            st.session_state.processed_upload_ids.add(upload_id)
            continue
        pending.append(item)

    if not pending:
        return

    with st.status("Analyzing multimodal uploads...", expanded=True) as status:
        for item in pending:
            st.write(f"Processing {item.name}")
            record = process_upload(item, use_ocr)
            st.session_state.processed_upload_ids.add(record.doc_id)
        status.update(label="Document intelligence refreshed", state="complete")
    st.rerun()


def render_home(docs: list[dict[str, Any]]) -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="status-pill">Live multimodal workspace</div>
            <div class="hero-title">NexusDoc AI</div>
            <div class="hero-copy">
                A research-grade document intelligence cockpit for PDFs, scans, images, invoices,
                resumes, handwritten notes, charts, and papers.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cards(docs)

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.subheader("Document Memory")
        if not docs:
            st.info("Upload files from the sidebar to begin analysis.")
        for doc in docs[:6]:
            analysis = doc.get("analysis", {})
            st.markdown(
                f"""
                <div class="doc-row">
                    <strong>{doc.get("name")}</strong>
                    <div class="small-muted">{analysis.get("doc_type")} · {analysis.get("word_count", 0):,} words · {doc.get("uploaded_at")}</div>
                    <div class="small-muted">{analysis.get("summary", "")[:220]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with right:
        st.subheader("Corpus Shape")
        if docs:
            type_counts = Counter(doc.get("analysis", {}).get("doc_type", "Document") for doc in docs)
            chart_df = pd.DataFrame({"Type": list(type_counts.keys()), "Documents": list(type_counts.values())})
            fig = px.bar(chart_df, x="Type", y="Documents", color="Type", template="plotly_dark")
            fig.update_layout(height=300, margin=dict(l=8, r=8, t=18, b=8), showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<div class="glass-panel small-muted">Charts appear after the first upload.</div>', unsafe_allow_html=True)


def render_analyzer(docs: list[dict[str, Any]]) -> None:
    st.subheader("Analyzer")
    if not docs:
        st.info("Upload a document to unlock preview, citations, insights, and extraction panels.")
        return
    names = {doc["name"]: doc["doc_id"] for doc in docs}
    selected_name = st.selectbox("Active document", list(names.keys()))
    doc = get_document(names[selected_name])
    if not doc:
        return
    analysis = doc.get("analysis", {})
    preview, intelligence = st.columns([.95, 1.25], gap="large")
    with preview:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.caption("Preview")
        image_preview(doc["path"], doc["file_type"])
        st.markdown("</div>", unsafe_allow_html=True)
    with intelligence:
        st.markdown("#### Intelligence Brief")
        st.write(analysis.get("summary", ""))
        if analysis.get("visual_summary"):
            with st.expander("Visual intelligence"):
                st.write(analysis["visual_summary"])
        st.markdown("".join(f'<span class="insight-chip">{term}</span>' for term in analysis.get("keywords", [])), unsafe_allow_html=True)
        cols = st.columns(3)
        cols[0].metric("Type", analysis.get("doc_type", "Document"))
        cols[1].metric("Read", f"{analysis.get('reading_minutes', 0)} min")
        cols[2].metric("Tone", analysis.get("tone", {}).get("label", "Neutral"))
        if st.button("Re-run OCR / vision analysis", use_container_width=True):
            with st.spinner("Refreshing document intelligence..."):
                doc = reprocess_document(doc, use_ocr=True)
            st.success("Document analysis refreshed.")
            st.rerun()
        if st.button("Generate report", use_container_width=True):
            path = make_report(doc)
            st.success(f"Report saved: {path.name}")

    sub_a, sub_b, sub_c, sub_d = st.tabs(["Extracted Text", "Citations", "Timeline", "Special Tools"])
    with sub_a:
        st.text_area("Text layer", value=doc.get("text", ""), height=360)
    with sub_b:
        citations = analysis.get("citations", [])
        if citations:
            for citation in citations:
                st.markdown(f"**Page {citation['page']}**: {citation['quote']}")
        else:
            st.info("No citations could be generated from the current text layer.")
    with sub_c:
        timeline = analysis.get("timeline", [])
        if timeline:
            timeline_df = pd.DataFrame(timeline)
            st.dataframe(timeline_df, use_container_width=True, hide_index=True)
        else:
            st.info("No explicit dates were detected.")
    with sub_d:
        tool = st.segmented_control(
            "Mode",
            ["Explain image + text", "Research simplifier", "Chart understanding", "Action items", "Emotional tone"],
        )
        if st.button("Run specialist analysis", type="primary"):
            prompt_map = {
                "Explain image + text": "Explain the visual and text evidence together. Include uncertainties and page references.",
                "Research simplifier": "Simplify this document for a smart non-specialist. Explain claims, methods, and limitations.",
                "Chart understanding": "Identify chart-like content, explain axes, trends, anomalies, and business meaning.",
                "Action items": "Extract concrete action items, owners if available, dates, and priorities.",
                "Emotional tone": "Detect emotional tone, persuasive stance, urgency, and confidence level.",
            }
            if tool in {"Explain image + text", "Chart understanding"} or doc.get("file_type") in IMAGE_TYPES:
                st.write(call_multimodal_llm(prompt_map[tool], doc))
            else:
                st.write(call_llm(prompt_map[tool], page_context(doc)))


def render_chat(docs: list[dict[str, Any]]) -> None:
    st.subheader("Chat Assistant")
    if not docs:
        st.info("Upload documents first, then ask cross-document questions.")
        return
    scope = st.radio("Context", ["All documents", "Selected document"], horizontal=True)
    selected_doc = None
    if scope == "Selected document":
        names = {doc["name"]: doc["doc_id"] for doc in docs}
        selected_doc = get_document(names[st.selectbox("Document", list(names.keys()), key="chat_doc")])
    history = chat_table.all()[-12:]
    for msg in history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    question = st.chat_input("Ask for summaries, citations, contradictions, risks, timelines, or next actions")
    if question:
        context = page_context(selected_doc) if selected_doc else "\n\n".join(page_context(doc) for doc in docs)
        if selected_doc and selected_doc.get("file_type") in IMAGE_TYPES:
            answer = call_multimodal_llm(question, selected_doc)
        else:
            answer = call_llm(question, context)
        chat_table.insert({"role": "user", "content": question, "created_at": datetime.now().isoformat(timespec="seconds")})
        chat_table.insert({"role": "assistant", "content": answer, "created_at": datetime.now().isoformat(timespec="seconds")})
        st.rerun()


def render_search(docs: list[dict[str, Any]]) -> None:
    st.subheader("Semantic Search")
    query = st.text_input("Search document memory", placeholder="Find revenue risks, applicant skills, citations about methodology...")
    if query:
        results = semantic_search(query, docs)
        if not results:
            st.warning("No matching passages found.")
        for item in results:
            st.markdown(
                f"""
                <div class="doc-row">
                    <strong>{item['document']} · Page {item['page']}</strong>
                    <div class="small-muted">Relevance {item['score']:.0%}</div>
                    <div>{item['snippet']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_compare(docs: list[dict[str, Any]]) -> None:
    st.subheader("Compare Documents")
    if len(docs) < 2:
        st.info("Upload at least two documents for side-by-side comparison.")
        return
    names = {doc["name"]: doc["doc_id"] for doc in docs}
    left_name, right_name = st.columns(2)
    with left_name:
        left_pick = st.selectbox("Left", list(names.keys()), key="left_compare")
    with right_name:
        right_pick = st.selectbox("Right", list(names.keys()), index=1, key="right_compare")
    left_doc = get_document(names[left_pick])
    right_doc = get_document(names[right_pick])
    if not left_doc or not right_doc:
        return
    comparison = compare_documents(left_doc, right_doc)
    st.metric("Concept similarity", f"{comparison['similarity']}%")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(f"#### {left_doc['name']}")
        st.write(comparison["left_summary"])
        st.caption("Unique signals")
        st.markdown("".join(f'<span class="insight-chip">{term}</span>' for term in comparison["left_only"]), unsafe_allow_html=True)
    with right:
        st.markdown(f"#### {right_doc['name']}")
        st.write(comparison["right_summary"])
        st.caption("Unique signals")
        st.markdown("".join(f'<span class="insight-chip">{term}</span>' for term in comparison["right_only"]), unsafe_allow_html=True)
    st.caption("Shared concepts")
    st.markdown("".join(f'<span class="insight-chip">{term}</span>' for term in comparison["common"]), unsafe_allow_html=True)


def render_resume_match(docs: list[dict[str, Any]]) -> None:
    st.subheader("Resume vs Job Description")
    resume_text = ""
    resume_docs = [doc for doc in docs if doc.get("analysis", {}).get("doc_type") == "Resume"]
    if resume_docs:
        options = {"Paste resume text": None} | {doc["name"]: doc["doc_id"] for doc in resume_docs}
        pick = st.selectbox("Resume source", list(options.keys()))
        if options[pick]:
            resume_text = get_document(options[pick]).get("text", "")
    pasted_resume = st.text_area("Resume text", value=resume_text, height=180)
    jd = st.text_area("Job description", height=180, placeholder="Paste the target role description...")
    if st.button("Match candidate", type="primary") and pasted_resume and jd:
        match = resume_match(pasted_resume, jd)
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=match["score"], title={"text": "Match Score"}, gauge={"axis": {"range": [0, 100]}}))
        gauge.update_layout(height=260, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, use_container_width=True)
        st.markdown("**Matched strengths**")
        st.markdown("".join(f'<span class="insight-chip">{term}</span>' for term in match["matched"]), unsafe_allow_html=True)
        st.markdown("**Missing or underrepresented**")
        st.markdown("".join(f'<span class="insight-chip">{term}</span>' for term in match["missing"]), unsafe_allow_html=True)


def render_reports(docs: list[dict[str, Any]]) -> None:
    st.subheader("Reports and Storage")
    st.caption(f"Storage directory: {STORAGE_DIR}")
    if docs:
        export = json.dumps(docs, indent=2)
        st.download_button("Download full memory JSON", export, "nexusdoc_memory.json", "application/json")
    reports = sorted(REPORT_DIR.glob("*_report.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not reports:
        st.info("Generate a report from the Analyzer tab.")
    for path in reports:
        with st.expander(path.name):
            st.code(path.read_text(encoding="utf-8"), language="json")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    inject_css()
    docs = get_documents()
    use_ocr, uploaded = render_sidebar(docs)
    render_upload_results(uploaded, use_ocr)

    tabs = st.tabs(["Dashboard", "Analyzer", "Chat", "Search", "Compare", "Resume Match", "Reports"])
    with tabs[0]:
        render_home(docs)
    with tabs[1]:
        render_analyzer(docs)
    with tabs[2]:
        render_chat(docs)
    with tabs[3]:
        render_search(docs)
    with tabs[4]:
        render_compare(docs)
    with tabs[5]:
        render_resume_match(docs)
    with tabs[6]:
        render_reports(docs)


if __name__ == "__main__":
    main()
