import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables from a .env file
load_dotenv()

# Define the working directory where the script is located
working_dir = os.path.dirname(os.path.abspath(__file__))


# Function to load a PDF document using UnstructuredPDFLoader
def load_document(file_path):
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()  # Load and parse the PDF file
    return documents


# Function to create a vector store for the document content
def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()  # Use HuggingFace for generating text embeddings
    text_splitter = CharacterTextSplitter(
        separator="/n",  # Split text using new line as separator
        chunk_size=1000,  # Set chunk size for document chunks
        chunk_overlap=200  # Allow some overlap between chunks
    )
    doc_chunks = text_splitter.split_documents(documents)  # Split the document into chunks
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)  # Create FAISS vector store from chunks
    return vectorstore


# Function to retrieve document chunks and generate a response with the LLM
def generate_rag_response(query, vectorstore, llm):
    # Retrieve relevant chunks from the document
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(query)  # Retrieve relevant chunks

    # Combine retrieved content to provide context for the language model
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Generate response with the LLM, using retrieved content as additional context
    response = llm.predict(f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
    return response


# Configure the Streamlit page
st.set_page_config(
    page_title="USCIS Chatbot",
    page_icon="☣️",
    layout="centered"
)

# Display the chatbot title and subtitle
st.title("☣️ Welcome to USCIS Chatbot")
st.subheader("Hi there 👋")

# Initialize chat history in Streamlit session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File uploader for user to upload a PDF document
uploaded_file = st.file_uploader(label="Upload USCIS Manual here 👇", type=["pdf"])

if uploaded_file:
    # Save the uploaded file to the working directory
    file_path = f"{working_dir}/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Setup vector store and conversational chain if not already initialized
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = setup_vectorstore(load_document(file_path))

    if "llm" not in st.session_state:
        st.session_state.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Define custom icons and styling for user and bot messages
user_icon = "♚"  # Set custom user icon emoji
bot_icon = "☣️"  # Set custom bot icon emoji
icon_size = 25  # Set the icon size in pixels

# Background color for user messages
user_background_color = "#f5f5f5"  # Light gray for user messages

# Loop to render the chat history in the UI with custom styling
for message in st.session_state.chat_history:
    if message["role"] == "user":
        # User message styling with icon inside the background
        icon_html = f"<span style='font-size:{icon_size}px; margin-right: 10px;'>{user_icon}</span>"
        background_style = f"background-color: {user_background_color}; padding: 10px; border-radius: 8px; display: flex; align-items: center; box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);"
        message_html = f"<div style='{background_style}'>{icon_html}{message['content']}</div>"

        # Display user message in a single block
        st.markdown(message_html, unsafe_allow_html=True)
    else:
        # Bot message styling with icon next to the message and left-aligned content
        icon_html = f"<span style='font-size:{icon_size}px; margin-right: 10px;'>{bot_icon}</span>"
        message_html = f"<div style='padding: 10px 0px; display: flex; align-items: flex-start;'>{icon_html}<div>{message['content']}</div></div>"

        # Display bot message with normal alignment and no special styling for list items
        st.markdown(message_html, unsafe_allow_html=True)

# Input field for user to ask a question
user_input = st.chat_input("Ask the Bot...")

if user_input:
    # Append user input to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Display user message with icon inside the background
    icon_html = f"<span style='font-size:{icon_size}px; margin-right: 10px;'>{user_icon}</span>"
    user_html = f"<div style='background-color: {user_background_color}; padding: 10px; border-radius: 8px; display: flex; align-items: center; box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);'>{icon_html}{user_input}</div>"
    st.markdown(user_html, unsafe_allow_html=True)

    # Generate RAG-based response using retrieved content
    response = generate_rag_response(user_input, st.session_state.vectorstore, st.session_state.llm)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display bot response with icon next to the text content, left-aligned
    icon_html = f"<span style='font-size:{icon_size}px; margin-right: 10px;'>{bot_icon}</span>"
    bot_html = f"<div style='padding: 10px 0px; display: flex; align-items: flex-start;'>{icon_html}<div>{response}</div></div>"
    st.markdown(bot_html, unsafe_allow_html=True)
