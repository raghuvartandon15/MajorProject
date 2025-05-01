import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("Groq API Key not found! Please set it in the .env file.")
    st.stop()

# Clear Chroma cache
chromadb.api.client.SharedSystemClient.clear_system_cache()

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.title("Finance & Tax Management AI Assistant")
st.write("Upload your financial or tax records (CSV) and ask any questions about them!")

# LLM initialization
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

# Session ID input
session_id = st.text_input("Session ID", value="finance_session")

# Chat history store
if "store" not in st.session_state:
    st.session_state.store = {}

# File upload
uploaded_files = st.file_uploader("Upload financial/tax CSV file(s)", type="csv", accept_multiple_files=True)

if uploaded_files:
    texts = []
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        for index, row in df.iterrows():
            row_text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
            texts.append(row_text)

    # Split and embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.create_documents(texts)

    vectorstore = Chroma.from_documents(split_docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Prompts
    summarization_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a finance and tax assistant. Summarize the following financial/tax records in clear, simple terms.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    advice_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant specialized in finance and taxes. Based on the provided CSV data, give practical advice in plain language. Use examples where helpful.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, summarization_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, advice_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # User input
    user_input = st.text_input("Ask a question (e.g., 'How much tax have I paid?', 'What are my major expenses?')")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input, "context": ""},
            config={"configurable": {"session_id": session_id}},
        )
        st.write("Assistant:", response["answer"])
        st.write("Chat History:", session_history.messages)
