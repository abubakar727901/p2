from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFProcessingError(Exception):
    """Raised when the permanent menu PDF cannot provide usable text."""


def extract_pdf_text(pdf_path):
    path = Path(pdf_path)
    if not path.is_file():
        raise PDFProcessingError(f"Menu PDF not found: {path.name}.")
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except (OSError, PdfReadError, ValueError) as exc:
        raise PDFProcessingError(
            "The menu PDF could not be read. Replace it with a valid, readable PDF."
        ) from exc
    text = "\n\n".join(page.strip() for page in pages if page.strip())
    if not text.strip():
        raise PDFProcessingError(
            "The menu PDF is empty or contains no extractable text. Use a text-readable PDF."
        )
    return text
