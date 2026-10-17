import re


def read_document(f) -> str:
    """Read an uploaded file (.pdf, .txt, .md) and return its text."""
    name = f.name.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(f)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return f.read().decode("utf-8", errors="ignore")


def clean_text(text: str) -> str:
    """Collapse extra spaces and line breaks into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_document(text: str, size: int = 450):
    """Split text into passages of roughly `size` characters, ending on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(text))
    chunks, current = [], ""
    for s in sentences:
        current = f"{current} {s}".strip()
        if len(current) >= size:
            chunks.append(current)
            current = ""
    if len(current) > 120:          # keep the last piece only if it is long enough
        chunks.append(current)
    return [c for c in chunks if len(c) > 120]   # drop tiny, useless passages