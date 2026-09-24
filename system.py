import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma-db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "jevaraat_products")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    if vectorstore._collection.count() == 0:
        raise RuntimeError("Chroma is empty. Run 'python create_db.py' first.")
    return vectorstore


def answer_question(query: str) -> str:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful Jevaraat jewellery assistant. Use only the provided context. "
                "Answer product questions using the exact values from the context. "
                "When a product matches, include its name and every available relevant field, "
                "especially category, product ID, metal, purity, weight, stone, size, price, "
                "making charge, description, and image URL. Never write 'no info' or omit a "
                "field that is present in the context. Do not invent missing values. "
                'If the answer is not present, say: "I could not find the answer in the product catalog."',
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion:\n{question}\n\n"
                "Give a clear Markdown answer with one product per section.",
            ),
        ]
    )
    model = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), temperature=0
    )
    response = model.invoke(prompt.invoke({"context": context, "question": query}))
    return str(response.content)
