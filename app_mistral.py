import streamlit as st
from langchain.llms import CTransformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader, TextLoader, DirectoryLoader
import time
import torch
import os

# Function to load documents
def load_documents(folder_path):
    start_time = time.time()
    with st.spinner("Loading documents..."):
        loader = DirectoryLoader(folder_path, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
        docs = loader.load()
    end_time = time.time()
    time_taken = end_time - start_time
    st.write(f"Loaded {len(docs)} documents in {time_taken:.2f} seconds")
    return docs

# Initialize language model and embeddings
# Replace the model path with the path to your locally downloaded model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with st.spinner("Initializing language model..."):
    model = CTransformers(model="mistral-7b-instruct-v0.1.Q4_K_M.gguf", model_type="mistral", max_new_tokens=1048, temperature=0.1, device=device)
with st.spinner("Initializing embeddings model..."):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs={'device': device}, encode_kwargs={'normalize_embeddings': False})

# Streamlit app
st.title("Document-based QA System")

# Folder path input
folder_path = st.text_input("Enter the path to the folder containing your documents")

if folder_path:
    # Check if vector store exists on disk
    if not os.path.exists("vector_store"):
        # Load documents
        docs = load_documents(folder_path)

        # Split documents into chunks
        start_time = time.time()
        with st.spinner("Splitting documents into chunks..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
            chunked_docs = splitter.split_documents(docs)
        end_time = time.time()
        time_taken = end_time - start_time
        st.write(f"Split {len(docs)} documents into {len(chunked_docs)} chunks in {time_taken:.2f} seconds")

        # Create vector store with the embeddings object
        start_time = time.time()
        with st.spinner("Creating vector store..."):
            db = FAISS.from_documents(chunked_docs, embeddings)
        end_time = time.time()
        time_taken = end_time - start_time
        st.write(f"Created vector store with {len(chunked_docs)} chunks in {time_taken:.2f} seconds")

        # Save vector store to disk
        with st.spinner("Saving vector store to disk..."):
            db.save_local("vector_store")

        st.success("Vector store created and saved to disk.")

    # Load vector store from disk
    start_time = time.time()
    with st.spinner("Loading vector store from disk..."):
        db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    end_time = time.time()
    time_taken = end_time - start_time
    st.write(f"Loaded vector store with {len(db.docstore._dict)} documents in {time_taken:.2f} seconds")

    # Create retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 2})

    # Create answer generation chain
    answer_generation_chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", return_source_documents=True, retriever=retriever)

    # Question input
    count = 0
    flag = True
    while flag:
        flag = False
        count += 1
        query = st.text_input(f"Question{count:}")

        if query:
            # Generate answer
            start_time = time.time()
            with st.spinner("Generating answer..."):
                response = answer_generation_chain(query)
                answer = response['result']
            end_time = time.time()
            time_taken = end_time - start_time
            st.write(f"Generated answer in {time_taken:.2f} seconds")

            # Display answer
            st.write(answer)
            flag = True
