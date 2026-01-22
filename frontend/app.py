import streamlit as st
import requests

st.set_page_config(page_title="Healthcare Assistant")
st.title("Healthcare Assistant")

query = st.text_input("Enter your question below:", "")
if st.button("Ask"):
    with st.spinner("Getting answer..."):
        res = requests.post("http://localhost:8000/ask", json={"question": query})
        answer = res.json().get("answer")
        st.success(answer)