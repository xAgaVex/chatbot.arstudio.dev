from langchain_chroma import Chroma
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import CHAT_MODEL, RETRIEVER_K

CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given the chat history and the latest user question, reformulate the "
    "question into a standalone question that can be understood without the "
    "chat history. Do NOT answer the question — just reformulate it if needed, "
    "otherwise return it as is."
)

QA_SYSTEM_PROMPT = (
    "You are an assistant answering questions about a collection of "
    "declassified US government documents (CIA, DOE, ODNI) concerning UFO/UAP "
    "sightings and related correspondence. The documents were OCR-scanned and "
    "the text is often garbled — interpret noisy text charitably.\n\n"
    "Answer ONLY from the provided context. If the answer is not in the "
    "context, say you couldn't find it in the documents — do not make "
    "anything up. Be concise and mention which document the information "
    "comes from when relevant.\n\n"
    "Context:\n{context}"
)


def build_chain(vectorstore: Chroma) -> Runnable:
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def ask(
    chain: Runnable, question: str, chat_history: list[BaseMessage]
) -> tuple[str, list[dict]]:
    """Run one conversational turn; return the answer and deduped sources."""
    result = chain.invoke({"input": question, "chat_history": chat_history})

    sources: list[dict] = []
    seen: set[tuple] = set()
    for doc in result.get("context", []):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_label") or doc.metadata.get("page", 0) + 1
        if (source, page) in seen:
            continue
        seen.add((source, page))
        sources.append(
            {"file": source, "page": page, "snippet": doc.page_content[:300]}
        )

    return result["answer"], sources
