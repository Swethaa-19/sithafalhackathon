import os
import requests
from bs4 import BeautifulSoup
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceLLM
from langchain.prompts import PromptTemplate
import streamlit as st

# Define the embedding model
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# Define the LLM model
llm_model = "t5-base"

# Define the vector database
vector_db = FAISS.from_embeddings(
    embeddings=HuggingFaceEmbeddings(model=embedding_model),
    dim=384,
)

# Crawl and scrape website content
def crawl_websites(websites):
    for website in websites:
        response = requests.get(website)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()
        yield text

# Convert text to embeddings and store in vector database
def store_embeddings(texts):
    embeddings = HuggingFaceEmbeddings(model=embedding_model)
    for text in texts:
        embedding = embeddings.encode(text)
        vector_db.add(embedding)

# Load the question-answering chain
def load_qa_chain(llm):
    prompt_template = """Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\nContext:\n{context}?\nQuestion:\n{question}\nAnswer:"""
    chain = load_qa_chain(llm, prompt_template, chain_type="stuff")
    return chain

# Define the chat function
def chat(query):
    llm = HuggingFaceLLM(model=llm_model)
    chain = load_qa_chain(llm)
    response = chain({"query": query}, return_only_outputs=True)
    return response

# Main function
def main():
    st.title("RAG Chatbot")
    websites = st.text_input("Enter URLs (separated by commas):")
    websites = [website.strip() for website in websites.split(",")]
    if st.button("Crawl and Chat"):
        texts = crawl_websites(websites)
        store_embeddings(texts)
        query = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            response = chat(query)
            st.write(response)

if __name__ == "__main__":
    main()
