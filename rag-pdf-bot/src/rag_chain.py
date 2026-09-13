"""
RAG chain: ties FAISS retrieval together with a Gemini LLM call using
a context-aware prompt template, and exposes a streaming interface
so the Streamlit UI can render tokens as they arrive.
"""

import os
from typing import Iterator, List, Tuple

from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-3.6-flash")

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the
context extracted from the user's uploaded PDF documents.

Rules:
- Answer strictly from the provided context.
- If the answer is not contained in the context, say you don't know rather
  than guessing.
- Be concise and cite which source/chunk supports each key claim when useful.

Context:
{context}

Conversation so far:
{history}

Question: {question}

Answer:"""


class RagChain:
    def __init__(self, vectorstore: FAISS, top_k: int = 4):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "history", "question"],
        )
        self.llm = ChatGoogleGenerativeAI(
            model=CHAT_MODEL,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            temperature=0.2,
            streaming=True,
        )
        self.history: List[Tuple[str, str]] = []

    def _retrieve(self, question: str):
        return self.vectorstore.similarity_search(question, k=self.top_k)

    def _format_context(self, docs) -> str:
        blocks = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "unknown")
            chunk_id = d.metadata.get("chunk_id", "?")
            blocks.append(f"[{i}] ({src}, chunk {chunk_id}):\n{d.page_content}")
        return "\n\n".join(blocks)

    def _format_history(self) -> str:
        if not self.history:
            return "(no previous turns)"
        return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in self.history[-3:])

    @staticmethod
    def _extract_text(content) -> str:
        """
        Normalize a streamed chunk's .content into plain text.

        Depending on the langchain-google-genai/langchain-core version,
        streaming content can arrive either as a plain string or as a
        list of content blocks (e.g. [{"type": "text", "text": "..."}]).
        """
        if not content:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    parts.append(block.get("text", ""))
            return "".join(parts)
        return str(content)

    def stream_answer(self, question: str) -> Iterator[Tuple[str, list]]:
        """
        Yields (text_chunk, sources) tuples. `sources` is populated only
        once retrieval has happened (first yielded alongside the stream)
        so the UI can render citations after streaming completes.
        """
        docs = self._retrieve(question)
        context = self._format_context(docs)
        history = self._format_history()

        final_prompt = self.prompt.format(
            context=context, history=history, question=question
        )

        sources = [
            {
                "source": d.metadata.get("source", "unknown"),
                "chunk": d.metadata.get("chunk_id", "?"),
                "text": d.page_content,
            }
            for d in docs
        ]

        full_answer = ""
        for chunk in self.llm.stream(final_prompt):
            token = self._extract_text(chunk.content)
            full_answer += token
            yield token, []

        # Sources are yielded once, after the text has fully streamed.
        yield "", sources

        self.history.append((question, full_answer))


def build_rag_chain(vectorstore: FAISS, top_k: int = 4) -> RagChain:
    return RagChain(vectorstore, top_k=top_k)
