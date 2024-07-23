import streamlit as st
from openai import OpenAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os

def load_documents(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.getvalue())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    
    os.unlink(temp_file_path)
    return docs

def get_vector_embedding(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    final_documents = text_splitter.split_documents(docs[:30])
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
    vector_store = FAISS.from_documents(final_documents, embeddings)
    return vector_store

st.set_page_config(page_title="Nvidia NIM Demo", layout="wide")
st.title("Nvidia NIM Demo")

with st.sidebar:
    nvidia_api_key = st.text_input("NVIDIA API KEY", key="NVIDIA_API_KEY", type="password")
    st.info("Please add your NVIDIA API key to continue.")

uploaded_file = st.file_uploader("Upload an article", type=["pdf"])
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

if uploaded_file and question and nvidia_api_key:
    with st.spinner("Processing your request..."):
        os.environ['NVIDIA_API_KEY']=nvidia_api_key
        docs = load_documents(uploaded_file)
        vector_store = get_vector_embedding(docs)

        prompt_template = ChatPromptTemplate.from_template(
            """
            Answer the question based on the provided context only.
            Please provide the most accurate response based on the question.
            
            Context:
            {context}
            
            Question: {input}
            """
        )

        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ['NVIDIA_API_KEY']
        )

        context = vector_store.similarity_search(question, k=5)
        context_text = "\n".join([doc.page_content for doc in context])
        prompt = prompt_template.format(context=context_text, input=question)

        try:
            completion = client.chat.completions.create(
                model="meta/llama3-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1024,
                stream=True
            )

            response = st.empty()
            full_response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    response.markdown(full_response + "▌")
            response.markdown(full_response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a PDF file, enter your question, and provide your NVIDIA API key to get started.")