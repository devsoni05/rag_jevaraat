import json
import os
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from database import fetch_product_documents

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma-db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "jevaraat_products")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


def product_to_document(item: dict[str, Any]) -> Document:
    collection_name = item["collection"]
    product = item["product"]
    product_id = str(product.get("_id", "unknown"))
    content = json.dumps(product, default=str, ensure_ascii=False, indent=2)

    return Document(
        page_content=(
            f"Product category: {collection_name}\n"
            f"Product ID: {product_id}\n"
            f"Product details:\n{content}"
        ),
        metadata={
            "source_collection": collection_name,
            "product_id": product_id,
        },
    )


def build_vector_database() -> int:
    records = fetch_product_documents()
    if not records:
        raise RuntimeError("No products were found in the configured MongoDB collections")

    documents = [product_to_document(record) for record in records]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    vectorstore.delete_collection()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    return len(chunks)


if __name__ == "__main__":
    chunk_count = build_vector_database()
    print(f"Indexed {chunk_count} chunks from MongoDB into {CHROMA_DIR}")

