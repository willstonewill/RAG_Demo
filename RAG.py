__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import os

st.set_page_config(page_title="RAG demo", layout="wide")

st.markdown("""
## Get insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging OpenAI model GPT-4o-mini. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### How It Works

Follow these simple steps to interact with the chatbot:

1. **Enter Your API Key**: You'll need a OpenAI API key for the chatbot to access GPT models.

2. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

3. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")



# This is the first API key input; no need to repeat it in the main function.
api_key = st.text_input("Enter your OpenAI API Key:", type="password", key="api_key_input")

def get_pdf_text(pdf_docs):
    
    documents = []
    for pdf in pdf_docs:
        file_name = "./" + pdf.name + ".pdf"
        with open(file_name, "wb") as file:
            file.write(pdf.getvalue())
            print(file_name)
        loader = PyMuPDFLoader(file_name)
        data = loader.load()
        documents.extend(data)
    return documents

def get_text_chunks(text):
    text_splitter = SemanticChunker(OpenAIEmbeddings(openai_api_key=api_key), add_start_index=True)
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embedding = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Chroma.from_documents(documents=text_chunks, embedding=embedding, persist_directory="./chroma_db")
    vectorstore.persist()

def format_docs(docs):
    return "\n\n".join(doc.page_content + f"\treport: {doc.metadata['source'].rsplit('/', 1)[-1].replace('.pdf', '')}" + f"\tpage: {doc.metadata['page']}" for doc in docs)

def user_input(user_question, api_key, chat_history):
    system_prompt = """
    Answer the question as detailed as possible from the provided context, include accurate document name and page number in the end in a new line, make sure to provide all the details, 
    if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Chat history: \n {chat_history}\n

    Answer:
    """
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    prompt =    (system_prompt)
    embedding = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough(), "chat_history": chat_history}
    | prompt
    | model
    | StrOutputParser()
    )

    response = ""
    for chunk in rag_chain.stream(user_question):
        response += chunk

    st.session_state["chat_answers_history"].append(response)
    st.session_state["user_prompt_history"].append(prompt)
    st.session_state["chat_history"].append((prompt,response))

    #st.write("Reply: ", response)

    if st.session_state["chat_answers_history"]:
       for i, j in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
          message1 = st.chat_message("user")
          message1.write(j)
          message2 = st.chat_message("assistant")
          message2.write(i)
    

def main():
    st.header("RAG chatbot💁")

    #user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    user_question = st.chat_input("Enter your questions here")
    if "user_prompt_history" not in st.session_state:
       st.session_state["user_prompt_history"]=[]
    if "chat_answers_history" not in st.session_state:
       st.session_state["chat_answers_history"]=[]
    if "chat_history" not in st.session_state:
       st.session_state["chat_history"]=[]

    if user_question and api_key:  # Ensure API key and user question are provided
        with st.spinner("Generating......"):
            user_input(user_question, api_key, chat_history = st.session_state["chat_history"])

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Done")

if __name__ == "__main__":
    main()