# USCIS Chatbot

A chatbot application that uses the **LLama 3.1: 8B model** to provide conversational responses based on uploaded USCIS manuals. It leverages a Retrieval-Augmented Generation (RAG) approach to fetch and process relevant sections from the manual, enabling accurate and contextually relevant responses to user queries.

## Setup Instructions

1. **Create a Virtual Environment**:
   ```bash
   python3 -m venv env
   source env/bin/activate   # On Windows, use `env\Scripts\activate`
   ```

2. **Install Dependencies**:
   Run the following command to install the necessary packages as specified in [requirements.txt](./requirements.txt):
   ```bash
   pip install -r requirements.txt
   ```
   You may add other dependencies specific to your environment or hardware (such as GPU support for `torch` if needed).

3. **Environment Variables**:
   - Create a `.env` file in the root directory to include any necessary API keys or environment variables, such as the **GROQ_API_KEY**. A template is provided in the [.env.example](./.env.example) file.


## Running the Application

1. **Start the Chatbot**:
   Run the following command to launch the application locally:
   ```bash
   streamlit run main.py
   ```

2. **Upload a Document**:
   Once the app is running, you can upload a USCIS manual (PDF) directly in the web interface. The chatbot will then be ready to answer your questions based on the document's content.

## Technology Overview

- **Streamlit**: A Python library for creating interactive web applications.
- **LangChain**: A framework for working with language models and document retrieval.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors, used for vector-based search.
- **HuggingFaceEmbeddings**: Utilized for generating vector embeddings from document content.
- **LLama 3.1: 8B Model**: The core language model powering the chatbot, designed for conversation and retrieval tasks.
- **UnstructuredPDFLoader**: A tool for loading PDF documents and preparing them for text processing.
- **Retrieval-Augmented Generation (RAG)**: The description now reflects the integration of RAG, which combines retrieved document content with generation for enhanced response relevance.

## Advanced Task Documentation

For integrating document contents as model weights, the approach involves:

1. **Extracting and Preprocessing Content**:
   - Use `UnstructuredPDFLoader` to extract content and `CharacterTextSplitter` to split text into manageable chunks.

2. **Generating Embeddings**:
   - Use `HuggingFaceEmbeddings` to convert text chunks into vector embeddings. The embeddings are then stored in FAISS for efficient retrieval.

3. **Integration with Chat Model**:
   - The LLama 3.1: 8B model, configured with `ChatGroq`, uses the embeddings to dynamically retrieve and respond with contextually relevant content using RAG.

4. **Methodology**:
   - The retrieval chain leverages FAISS and embeddings to provide real-time, document-aware responses by combining retrieved content with language generation. This ensures that the chatbot remains relevant and informative by focusing on specific content from the uploaded document.

## Additional Enhancements

- **Custom Icons and Message Styling**: The chat interface includes unique icons and stylings for user and bot messages, improving the user experience.
- **Explore Optional Model Management with Ollama**: We plan to integrate Ollama for more efficient model deployment and serving. This will streamline model handling and offer additional configuration options for production environments.
- **Create Advanced Model with Preloaded Content as Weights**: We aim to implement an advanced feature of preloading content as model weights. This involves encoding the content using embeddings, then modifying the LLM architecture to incorporate these embeddings as part of the model's weights. We will also document the methodology, detailing the steps for integrating the embeddings with the model and any considerations for deploying this advanced model.
