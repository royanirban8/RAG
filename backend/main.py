from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import get_rag_chain
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_chain = get_rag_chain()


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask_question(query: Query):
    result = rag_chain.run(query.question)
    return {"answer": result}
