import os
import glob
from pathlib import Path
from typing import Dict, Any, List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)

llm_model = "qwen/qwen3.8-27b"

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def fetch_documents():
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            src = doc.metadata.get("source", "")
            doc.metadata["doc_type"] = doc_type
            doc.metadata["file_name"] = os.path.basename(src)
            documents.append(doc)
    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        try:
            Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()
        except Exception as e:
            print(f"Notice during collection cleanup: {e}")

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore, count, dimensions


def run_ingestion() -> Dict[str, Any]:
    """Execute complete ingestion pipeline and return summary stats."""
    docs = fetch_documents()
    chunks = create_chunks(docs)
    vectorstore, count, dimensions = create_embeddings(chunks)
    return {
        "status": "success",
        "documents": len(docs),
        "chunks": len(chunks),
        "vectors": count,
        "dimensions": dimensions,
    }


if __name__ == "__main__":
    result = run_ingestion()
    print("Ingestion complete:", result)

