import time
from collections.abc import Callable

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DOCS_DIR,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    MIN_PAGE_CHARS,
)

ProgressCallback = Callable[[str, int, int], None]


def load_documents(
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[Document], int]:
    """Load all PDFs from DOCS_DIR, dropping pages with no usable text.

    Returns the kept pages and the number of skipped (near-empty) pages.
    """
    pdf_paths = sorted(DOCS_DIR.glob("*.pdf"))
    documents: list[Document] = []
    skipped = 0
    for i, path in enumerate(pdf_paths, start=1):
        if progress_callback:
            progress_callback(path.name, i, len(pdf_paths))
        for doc in PyPDFLoader(str(path)).load():
            if len(doc.page_content.strip()) < MIN_PAGE_CHARS:
                skipped += 1
                continue
            doc.metadata["source"] = path.name
            documents.append(doc)
    return documents, skipped


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


class _Embeddings(GoogleGenerativeAIEmbeddings):
    """Coerce inputs to plain str: the google-genai client 500s on str
    subclasses like langchain-core's TextAccessor (produced by
    StrOutputParser inside create_history_aware_retriever)."""

    def embed_query(self, text: str, **kwargs) -> list[float]:
        return super().embed_query(str(text), **kwargs)

    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        return super().embed_documents([str(t) for t in texts], **kwargs)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return _Embeddings(model=EMBEDDING_MODEL)


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def _add_with_retry(
    vectorstore: Chroma, batch: list[Document], attempts: int = 5
) -> None:
    """Add a batch, waiting out free-tier 429s (100 embed requests/min)."""
    for attempt in range(attempts):
        try:
            vectorstore.add_documents(batch)
            return
        except Exception as e:
            if "RESOURCE_EXHAUSTED" not in str(e) or attempt == attempts - 1:
                raise
            time.sleep(65)


def ingest(
    file_callback: ProgressCallback | None = None,
    batch_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Rebuild the vector store from the PDFs in DOCS_DIR."""
    documents, skipped = load_documents(file_callback)
    chunks = split_documents(documents)

    vectorstore = get_vectorstore()
    vectorstore.reset_collection()
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        _add_with_retry(vectorstore, chunks[start : start + EMBED_BATCH_SIZE])
        if batch_callback:
            batch_callback(min(start + EMBED_BATCH_SIZE, len(chunks)), len(chunks))

    return {
        "files": len(sorted(DOCS_DIR.glob("*.pdf"))),
        "pages": len(documents),
        "skipped_pages": skipped,
        "chunks": len(chunks),
    }


def chunk_count() -> int:
    """Number of chunks in the persisted store; 0 if it doesn't exist yet."""
    if not CHROMA_DIR.exists():
        return 0
    try:
        return get_vectorstore()._collection.count()
    except Exception:
        return 0


def vectorstore_exists() -> bool:
    return chunk_count() > 0
