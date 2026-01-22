from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

DB_FAISS_PATH = "vectorstore/db_faiss"

#create a prompt template
template = """
Your name is Mark. You are a Data Structures Tutor. Your task is to answer any questions related to this field.
Provide a concise answer to the following question using relevant information provided below and your 
understanding of the question.

Relevant information:
{context}

If the information above does not answer the question below, try to answer on your own if you are confident enough or else 
say "I don't know the answer to this.".

Question:
{question}

 <|end|>
"""

#define prompt template
prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"])

def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 6})

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512, device="mps")
    llm = HuggingFacePipeline(pipeline=pipe)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    return chain
