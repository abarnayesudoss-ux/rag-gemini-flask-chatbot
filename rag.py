import io
import os
import sqlite3
import math
from pypdf import PdfReader
from docx import Document

DB_PATH = os.path.join(os.path.dirname(__file__), "rag_store.db")
EMBEDDING_MODEL = "gemini-embedding-001"
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}

def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    """)
    return conn

def _extract_text(file):
    name = file.filename or ""
    ext = os.path.splitext(name)[1].lower()
    raw = file.read()

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Supported files: PDF, DOCX and TXT.")

    if ext == ".txt":
        return raw.decode("utf-8", errors="ignore")

    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    doc = Document(io.BytesIO(raw))
    return "\n".join(p.text for p in doc.paragraphs)

def _chunks(text, size=1200, overlap=200):
    text = " ".join(text.split())
    if not text:
        return []

    result = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        result.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return result

def _embed(client, text):
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )
    return response.embeddings[0].values

def add_document(file, client):
    text = _extract_text(file)
    chunks = _chunks(text)

    if not chunks:
        raise ValueError("The document does not contain readable text.")

    conn = _db()
    try:
        for chunk in chunks:
            vector = _embed(client, chunk)
            conn.execute(
                "INSERT INTO chunks(source, content, embedding) VALUES (?, ?, ?)",
                (file.filename, chunk, ",".join(map(str, vector)))
            )
        conn.commit()
    finally:
        conn.close()

    return len(chunks)

def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

def retrieve_context(question, client, top_k=5):
    query_vector = _embed(client, question)
    conn = _db()

    try:
        rows = conn.execute(
            "SELECT source, content, embedding FROM chunks"
        ).fetchall()
    finally:
        conn.close()

    scored = []
    for source, content, embedding in rows:
        vector = [float(x) for x in embedding.split(",")]
        scored.append((_cosine(query_vector, vector), source, content))

    scored.sort(reverse=True, key=lambda item: item[0])
    selected = scored[:top_k]

    return "\n\n".join(
        f"[Source: {source}]\n{content}" for _, source, content in selected
    )
