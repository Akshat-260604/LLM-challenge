import streamlit as st
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers

# Helper functions
def load_pdf(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# Streamlit UI
st.title("PDF Question Answering with Page References")
st.write("""
    This app allows you to ask questions from PDF documents stored in the **data/** folder. 
    The answers include the relevant context and page references.
""")

# Load PDFs and preprocess
st.header("Loading and Preprocessing Documents")
with st.spinner("Loading PDF files and preparing chunks..."):
    extracted_data = load_pdf("data/")
    text_chunks = text_split(extracted_data)
    embeddings = download_hugging_face_embeddings()
    docsearch = FAISS.from_texts([t.page_content for t in text_chunks], embeddings)
st.success("PDFs loaded and processed successfully!")

# Prompt template
prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Initialize LLM
llm = CTransformers(
    model="shrestha-prabin/llama-2-7b-chat.ggmlv3.q4_0",
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Question input
st.header("Ask a Question")
user_input = st.text_input("Enter your question here:")

if user_input:
    with st.spinner("Fetching the answer..."):
        result = qa({"query": user_input})
        response = result["result"]
        source_documents = result["source_documents"]

    # Display answer
    st.subheader("Answer:")
    st.write(response)

    # Display source information
    st.subheader("Source Documents:")
    for doc in source_documents:
        page_ref = doc.metadata.get("page", "Unknown")
        context_preview = doc.page_content[:300]  # Show first 300 characters
        st.write(f"**Page {page_ref}:** {context_preview}...")

    st.success("Answer and references displayed successfully!")
