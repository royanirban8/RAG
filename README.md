# Healthcare RAG Chatbot

This is an open-source Retrieval-Augmented Generation (RAG) chatbot for answering medical queries using LangChain, Hugging Face, FastAPI, and Streamlit.

## How to Run
1. Put PDFs in `data/`
2. Generate vector DB:
```bash
cd backend
python document_loader.py
```
3. Start services:
```bash
docker-compose up --build
```
4. Visit `http://localhost:8501` to use the chatbot.

## License: Apache 2.0
```