"""
Vector store utilities: embed document chunks with Gemini embeddings
and build/persist a FAISS similarity index.
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

EMBEDDING_MODEL = "gemini-embedding-001"


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set. Add it in the sidebar.")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)


def build_vectorstore(chunks: List[Document]) -> FAISS:
    embeddings = _get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def save_vectorstore(vectorstore: FAISS, path: str) -> None:
    vectorstore.save_local(path)


def load_vectorstore(path: str) -> FAISS:
    embeddings = _get_embeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
