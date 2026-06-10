# UAP Document Chatbot

A chatbot that answers questions about declassified US government UFO/UAP documents
(CIA, DOE, ODNI). Ask it something, it digs through the scanned PDFs and answers
with citations — file name and page number included.

**Live demo:** [www.chatbot.arstudio.dev](https://www.chatbot.arstudio.dev)

## How it works

The PDFs get split into ~1000-character chunks, embedded with Gemini
(`gemini-embedding-001`) and stored in a local ChromaDB. When you ask a question,
the most relevant chunks are retrieved and passed to `gemini-3.5-flash`, which
answers only from those documents — if it's not in the files, it says so.

It also handles follow-ups: a question like *"did it make any sound?"* is first
rewritten into a standalone question using the chat history, so retrieval still
finds the right pages.

Stack: Python, LangChain, ChromaDB, Gemini API, Streamlit.

## Try asking

- *What did the senior intelligence officer encounter during the helicopter flight in 2025?*
- *What did a source in the USSR observe at Site 7 in 1973?* — then follow up with *"did it make any sound?"*
- *Who was Dr. Lincoln LaPaz and what did he investigate?*
- *What were the green fireballs seen over New Mexico in the late 1940s?*
- *What happened at the Pantex plant?*

## Run it locally

```bash
git clone <this repo>
cd chatbot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # put your Gemini API key here (free at aistudio.google.com/apikey)
streamlit run app.py
```

Then click **Process documents** in the sidebar (builds the vector store, takes a
few minutes on the free tier) and start asking. The store is persisted in
`chroma_db/`, so it's a one-time thing.

## A note on the documents

These are real declassified scans, and the OCR is rough in places — you'll see
garbled text in the cited snippets. That's the source material, not a bug.
Pages with no readable text at all are skipped during processing.
