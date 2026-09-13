"""
Document ingestion: load multiple PDFs and split them into overlapping
text chunks suitable for embedding.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def load_and_chunk_pdfs(
    pdf_paths: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Load one or more PDF files and split their contents into chunks.

    Each resulting chunk keeps metadata about which file and page it
    came from, which is later surfaced to the user as a citation.
    """
    all_docs: List[Document] = []

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = _basename(path)
        all_docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    # Tag each chunk with a stable index per source file for citation display.
    counters = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        counters[src] = counters.get(src, 0) + 1
        chunk.metadata["chunk_id"] = counters[src]

    return chunks


def _basename(path: str) -> str:
    import os

    return os.path.basename(path)
