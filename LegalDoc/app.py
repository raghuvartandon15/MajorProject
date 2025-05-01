import os
import streamlit as st
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
from langchain_community.document_loaders import PyPDFLoader
import chromadb

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Ensure API key is loaded
if not groq_api_key:
    st.error("Groq API Key not found! Please set it in the .env file.")
    st.stop()

# Clear system cache
chromadb.api.client.SharedSystemClient.clear_system_cache()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.title("Legal Document Adviser")
st.write("Upload a legal document to get a simplified summary and advice.")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

# Session ID input
session_id = st.text_input("Session ID", value="default_session")

# Manage chat history state
if "store" not in st.session_state:
    st.session_state.store = {}

# PDF Uploads
uploaded_files = st.file_uploader("Upload a legal document (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temp_pdf = "./temp.pdf"
        with open(temp_pdf, "wb") as file:
            file.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        documents.extend(docs)

    # Split text and create embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    
    # Extract raw text from document splits
    texts = [doc.page_content for doc in splits]  

    # Ensure embeddings are created from text, not documents
    vectorstore = Chroma.from_texts(texts=texts, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # System prompts for summarization
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize the following legal document in simple, easy-to-understand terms.\n\nContext:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # System prompts for legal advice
    advice_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a legal assistant. Based on the given legal document, provide advice in plain language.\n\nContext:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

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

    # User input for summary or advice
    user_input = st.text_input("Ask about the document (e.g., 'Summarize this' or 'What does this mean for me?')")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input, "context": ""},  # Ensure context is passed
            config={"configurable": {"session_id": session_id}},
        )
        st.write("Assistant:", response["answer"])
        st.write("Chat History:", session_history.messages)
