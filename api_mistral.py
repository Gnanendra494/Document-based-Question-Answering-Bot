import streamlit as st
import os
import requests
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pickle

# Function to create a bordered chat message
def bordered_chat_message(role, content):
    color = "#AED9E0" if role == "user" else "#D5E8ED"
    st.markdown(f"""
        <div style="
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            background-color: {color};
            color: #000000;
            word-wrap: break-word;
            white-space: pre-wrap;
            word-break: break-word;
            font-family: Helvetica;
        ">
        <strong>{role.capitalize()}:</strong> {content}
        </div>
    """, unsafe_allow_html=True)

# Function to read PDFs
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to process documents
def process_documents(folder_path):
    vector_store_path = "vector_store_api"
    if os.path.exists(vector_store_path):
        # Load existing vector store
        with open(vector_store_path, "rb") as f:
            vectorstore = pickle.load(f)
        st.info("Loaded existing vector store.")
    else:
        documents = []
        for file in os.listdir(folder_path):
            if file.endswith('.pdf'):
                file_path = os.path.join(folder_path, file)
                text = read_pdf(file_path)
                documents.append(text)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.create_documents(documents)

        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(texts, embeddings)

        # Save vector store
        with open(vector_store_path, "wb") as f:
            pickle.dump(vectorstore, f)
        st.success("Created and saved new vector store.")

    return vectorstore

# Mistral API call function
def mistral_api_call(prompt, max_tokens=100):
    api_key = "wzbZ3aJNMgwJe0j3wkc43UoviXoPxsd2 "  # Replace with your actual API key
    url = "https://api.mistral.ai/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": "mistral-tiny",  # or whichever model you're using
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()['choices'][0]['message']['content']


# Create a container for the sidebar
sidebar = st.sidebar

# Streamlit app
sidebar.title("Document Q&A Chatbot")

# Folder path input in the sidebar
folder_path = sidebar.text_input("Enter the folder path containing PDF files:")

if folder_path:
    if "vectorstore" not in st.session_state:
        with st.spinner("Processing documents..."):
            st.session_state.vectorstore = process_documents(folder_path)

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Create a container for chat history
    chat_container = st.container()

    # Display chat history in the chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    with chat_container:
        for message in st.session_state.messages:
            bordered_chat_message(message["role"], message["content"])
    st.markdown('</div>', unsafe_allow_html=True)

    # User input and submit button
    input_container = st.container()
    with input_container:
        col1, col2 = st.columns([6, 1])  # Adjust the ratio as needed
        with col1:
            prompt = st.text_input("What is your question?", key="user_input", label_visibility="collapsed")
        with col2:
            send_button = st.button("Send")

    # Custom CSS to style the input, button, and chat container
    st.markdown("""
        <style>
        .stTextInput > div > div > input {
            padding-right: 100px;
        }
        .stButton {
            position: absolute;
            top: 0;
            right: 0;
            height: 100%;
        }
        .stButton > button {
            height: 100%;
            padding-top: 0;
            padding-bottom: 0;
        }
        .chat-container {
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
            max-height: 400px;
            overflow-y: auto;
        }
        .input-container {
            position: fixed;
            border: 2px solid #e0e0e0;
            bottom: 0;
            left: 0;
            height: 140px;
            width: 100%;
            background-color: #f0f2f6;
            padding: 10px;
            border-top: 1px solid #e0e0e0;
        }
        </style>
        """, unsafe_allow_html=True)

    if send_button and prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_container:
            bordered_chat_message("user", prompt)

            with st.spinner("Thinking..."):
                # Retrieve relevant context
                retriever = st.session_state.vectorstore.as_retriever()
                docs = retriever.get_relevant_documents(prompt)
                context = "\n".join([doc.page_content for doc in docs])

                # Formulate prompt for Mistral
                full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
                response = mistral_api_call(full_prompt)

                st.session_state.messages.append({"role": "assistant", "content": response})
                bordered_chat_message("assistant", response)

        # Clear the input field after sending
        st.experimental_rerun()

else:
    st.warning("Please enter a folder path to start.")
