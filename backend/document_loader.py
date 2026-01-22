import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DATA_DIR = "../data"
DB_FAISS_PATH = "vectorstore/db_faiss"


def build_vectorstore():
    docs = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, file))
            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        texts = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    build_vectorstore()
