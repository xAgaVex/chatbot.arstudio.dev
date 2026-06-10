import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "uap_docs"

CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 4
MIN_PAGE_CHARS = 20
EMBED_BATCH_SIZE = 100


def require_api_key() -> str | None:
    """Return the Gemini API key, or None with no side effects if unset."""
    return os.getenv("GOOGLE_API_KEY") or None
