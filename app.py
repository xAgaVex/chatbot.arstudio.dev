import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.config import DOCS_DIR, require_api_key
from src.ingestion import chunk_count, get_vectorstore, ingest, vectorstore_exists
from src.rag_chain import ask, build_chain

st.set_page_config(page_title="UAP Document Chatbot", page_icon="🛸", layout="wide")

if not require_api_key():
    st.error(
        "Missing `GOOGLE_API_KEY`. Copy `.env.example` to `.env` and add your "
        "Gemini API key, then restart the app."
    )
    st.stop()


@st.cache_resource(show_spinner="Setting up the document index (first run only)...")
def load_chain():
    if not vectorstore_exists():
        ingest()
    return build_chain(get_vectorstore())


if "messages" not in st.session_state:
    st.session_state.messages = []

# Builds the index on the very first request after a cold start (chroma_db/
# is gitignored, so it doesn't ship with the deploy). Cached afterwards.
chain = load_chain()

with st.sidebar:
    st.header("📄 Documents")
    pdf_paths = sorted(DOCS_DIR.glob("*.pdf"))
    if pdf_paths:
        for path in pdf_paths:
            st.caption(f"• {path.name}")
    else:
        st.warning(f"No PDFs found in `{DOCS_DIR.name}/`.")

    count = chunk_count()
    if count:
        st.success(f"Vector store ready ({count} chunks)")
    else:
        st.info("Vector store not built yet — process the documents below.")

    if st.button("Process documents", disabled=not pdf_paths):
        with st.status("Processing documents...", expanded=True) as status:
            file_progress = st.progress(0.0, text="Loading PDFs...")
            embed_progress = st.progress(0.0, text="Embedding...")

            def on_file(name: str, i: int, total: int) -> None:
                file_progress.progress(i / total, text=f"Loading {name} ({i}/{total})")

            def on_batch(done: int, total: int) -> None:
                embed_progress.progress(
                    done / total, text=f"Embedding chunks ({done}/{total})"
                )

            stats = ingest(file_callback=on_file, batch_callback=on_batch)
            load_chain.clear()
            status.update(label="Done", state="complete")
        st.success(
            f"Indexed {stats['chunks']} chunks from {stats['pages']} pages "
            f"across {stats['files']} files "
            f"({stats['skipped_pages']} near-empty pages skipped)."
        )
        st.rerun()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("🛸 UAP Document Chatbot")
st.caption(
    "Ask questions about the declassified CIA/DOE/ODNI UAP documents. "
    "Answers cite their sources."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for src in message["sources"]:
                    st.markdown(f"**{src['file']}** — page {src['page']}")
                    st.caption(src["snippet"])

if question := st.chat_input("Ask about the documents..."):
    with st.chat_message("user"):
        st.markdown(question)

    chat_history = []
    for message in st.session_state.messages:
        if message["role"] == "user":
            chat_history.append(HumanMessage(content=message["content"]))
        else:
            chat_history.append(AIMessage(content=message["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = ask(chain, question, chat_history)
            except Exception:
                st.error(
                    "Gemini is rate-limited or temporarily unavailable. "
                    "Please wait a moment and try again."
                )
                st.stop()
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.markdown(f"**{src['file']}** — page {src['page']}")
                    st.caption(src["snippet"])

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
