from fastapi import FastAPI
from pydantic import BaseModel

from system import answer_question

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "Jevaraat RAG server is running"}


@app.post("/ask")
def ask_question(data: QueryRequest):
    return {"answer": answer_question(data.query)}