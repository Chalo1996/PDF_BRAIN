from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter as cs
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from templates.template import css, bot_template, user_template
from typing import List


load_dotenv()

def get_pdf_text(pdf_docs):
    """
    get_pdf_text Extracts text from a PDF document.

    Args:
        pdf_docs: A list of PDF documents

    Returns:
        text: A string containing the text from the PDF documents
    """
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
            text += "\n"
    return text


def get_text_chunks(raw_text: str) -> List[str]:
    """
    get_text_chunks Splits the text into chunks.

    Args:
        raw_text: A string containing the text from the PDF documents

    Returns:
        chunks: A list of strings containing the text chunks
    """
    # Splitting the text into chunks of 1000 characters with 100 overlap.
    # The length function is used to calculate the length of each chunk.
    # The separator is used to split the text into chunks.
    # The text is split into chunks of 1000 characters with 100 overlap.
    # The length function is used to calculate the length of each chunk.
    # The separator is used to split the text into chunks.
    text_splitter = cs(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        separator="\n"
    )
    return text_splitter.split_text(raw_text)


def get_vector_storefromOpenAI(chunks):
    """
    get_vector_storefromOpenAI Creates Embeddings using OpenAI\
        model and stores them in a vector store.

    This is not Open Source and is expensive.

    Args:
        chunks: A list of strings extracted from the PDF

    Returns:
    vector_store: A vector store containing the embeddings
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
        )
    return vectorstore


def get_vector_storefromIntructor(chunks):
    """
    get_vector_storefromIntructor Creates Embeddings using Intructor\
        model and stores them in a vector store.

    This is an Open Source and is Free. The embeddings are stored\
        locally and are also created using your own machine resources.\
            It is also slower as compared to the OpenAI embedding module\
                because it uses your computer's GPU and CPU.

    Args:
        chunks: A list of strings extracted from the PDF

    Returns:
    vector_store: A vector store containing the embeddings
    """
    embedding = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
    )
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embedding
        )
    return vectorstore


def get_conversation_chain(vector_store):
    """
    get_conversation_chain Creates a conversation chain from the vector store.

    Args:
        vector_store: A vector store containing the embeddings

    Returns:
    conversation_chain: A conversation chain
    """
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_user_question(user_question: str):
    """
    handle_user_question Handles the user question.

    Args:
        user_question: A string containing the user question

    Returns:
    response: A dictionary containing the response
    """
    if st.session_state.conversation is None:
        st.session_state.conversation = get_conversation_chain(
            get_vector_storefromIntructor(get_text_chunks(get_pdf_text(st.session_state.pdf_docs)))
        )
    st.session_state.chat_history = []
    response = st.session_state.conversation(
        {"question" : user_question})
    st.session_state.chat_history = response['chat_history']

    for idx, msg in enumerate(st.session_state.chat_history):
        if idx % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)


def main():
    st.set_page_config(
    page_title="PDF Brain",
    page_icon=":books:",
    )

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.markdown("# Welcome to the **ExRx** PDF to Chart App Engine.")
    st.header("The PDF (:books:) Brain!")
    user_question = st.text_input("Ask question about your document/s here.")

    if user_question:
        with st.spinner("Processing..."):
            handle_user_question(user_question)

    with st.sidebar:
        st.subheader("Choose PDFs")
        st.write("Choose the PDFs you want to use in your chat.")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        if st.button("Upload PDFs"):
            with st.spinner("Uploading..."):
                raw_text: str = get_pdf_text(pdf_docs)
                chunks =  get_text_chunks(raw_text)
                vector_store = get_vector_storefromOpenAI(chunks)
                st.session_state.conversation = \
                    get_conversation_chain(vector_store)


if __name__ == "__main__":
    main()
