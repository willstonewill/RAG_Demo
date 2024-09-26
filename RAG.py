# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
#from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import RunnableLambda
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

# def get_session_id():
#     if 'session_id' not in st.session_state:
#         # Generate a new UUID and store it in session_state
#         st.session_state['session_id'] = str(uuid.uuid4())
#     return st.session_state['session_id']

# session_id = get_session_id()

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
        os.remove(file_name)
    return documents

def get_text_chunks(text):
    text_splitter = SemanticChunker(OpenAIEmbeddings(model="gpt-4o-2024-08-06",openai_api_key=api_key), add_start_index=True)
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embedding = OpenAIEmbeddings(model="gpt-4o-2024-08-06", openai_api_key=api_key)
    # vectorstore = Chroma.from_documents(documents=text_chunks, embedding=embedding, persist_directory="./chroma_db")
    # vectorstore.persist()
    
    # Create and store the vectorstore in session_state
    vectorstore = FAISS.from_documents(text_chunks, embedding=embedding)
    st.session_state['vectorstore'] = vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content + f"\treport: {doc.metadata['source'].rsplit('/', 1)[-1].replace('.pdf', '')}" + f"\tpage: {doc.metadata['page']}" for doc in docs)


def user_input(user_question, api_key, chat_history):
    system_prompt = system_prompt = """
    You are an AI assistant that provides detailed answers based only on two parts: the provided context and the chat history. Your goal is to help the user by answering their question as comprehensively as possible, using information from both the context and the chat history.

    Instructions:
    - **Utilize Both Context and Chat History**: Always consider both when formulating your answer. Answer may exist in the graphs or tables in the document.
    - **Comparative Questions**: For comparison tasks, provide clear, structured answers referencing relevant documents, names, and page numbers.
    - **Detail Orientation**: Include all pertinent details from both sources to fully answer the question.
    - **Unavailable Information**: If the answer isn't in the context or chat history, respond with "Answer is not available in the provided context."
    - **Accuracy**: Only use information present in the context or chat history. Do not speculate.
    - **Formatting**: Use bullet points, tables, graphs or paragraphs for clarity.

    Context:
    {context}

    Chat History:
    {chat_history}

    Question:
    {question}

    Answer:
    """


    model = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=1, api_key=api_key)
    prompt = ChatPromptTemplate.from_template(system_prompt)
    embedding = OpenAIEmbeddings(openai_api_key=api_key)
    #vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
    if 'vectorstore' in st.session_state:
    #vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
        vectorstore = st.session_state['vectorstore']
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    else:
        st.write("Pleas make sure you uploaded pdf files and clicked sumbit! And make sure the file size is not too large.")

    chat_history_runnable = RunnableLambda(lambda _: st.session_state.get("chat_history", ""))


    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough(), "chat_history": chat_history_runnable}
    | prompt
    | model
    | StrOutputParser()
    )

    response = ""
    for chunk in rag_chain.stream(user_question):
        response += chunk

    #st.write(f'"Chat History:", {st.session_state.get("chat_history", [])}')

    st.session_state["chat_answers_history"].append(response)
    st.session_state["user_prompt_history"].append(user_question)
    st.session_state['chat_history'].append(f"User: {user_question}")
    st.session_state['chat_history'].append(f"Assistant: {response}")

    #st.write("Reply: ", response)

    if st.session_state["chat_answers_history"]:
       for i, j in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
          message1 = st.chat_message("user")
          message1.write(j)
          message2 = st.chat_message("assistant")
          message2.write(i)
    
def reset_conversation():
  st.session_state["chat_answers_history"] = []
  st.session_state["user_prompt_history"] = []
  st.session_state["chat_history"] = []

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

    st.button('Reset Chat', on_click=reset_conversation)

if __name__ == "__main__":
    main()