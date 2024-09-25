## Introduction

This repository provides a system that integrates a knowledge base with a chatbot interface, leveraging a retrieval-augmented generation (RAG) architecture. The system is built using `LangChain` for document processing and retrieval, `Chroma` as the vector database for storing and retrieving document embeddings, and the OpenAI API for generating responses from a language model (LLM). It also incorporates a reranking mechanism using a sentence transformer model to prioritize the most relevant documents, ensuring accurate and context-aware responses to user queries.

## Features

- **Knowledge Base Creation**: 
  - The `build_knowledge_base.py` script processes markdown and PDF documents (Dengue Fever related in this project), splits them into smaller chunks, and stores them as embeddings in a vector database using OpenAI embeddings and the Chroma vector store.
  
- **Querying the Knowledge Base**: 
  - The `query.py` script allows users to interact with the knowledge base via command-line queries. It retrieves relevant documents using Chroma, reranks them for relevance, and generates a response using OpenAI's language models.
  
- **Gradio Chatbot Interface**: 
  - The `gradio_query.py` script provides a user-friendly chatbot interface using Gradio. Users can submit questions, and the system will generate detailed answers along with sources from the knowledge base.

## How It Works

<img src=".\images\arch.png" alt="arch" style="zoom:40%;" />

1. **Build the Knowledge Base**:
   - Run the `build_knowledge_base.py` script to load markdown and PDF documents from the specified directories.
   - The script will chunk, embed, and store the documents in a Chroma vector database for efficient retrieval.

2. **Query the Knowledge Base**:
   - The `query.py` script allows users to query the knowledge base from the command line. The system will retrieve the most relevant documents and use a language model to generate responses based on the content.

3. **Gradio Chatbot**:
   - The `gradio_query.py` script provides an interactive web interface for querying the knowledge base. Users can type in questions, and the chatbot will respond with answers and document sources.

## Example output

<img src=".\images\example1.png" alt="arch" />

<img src=".\images\example2.png" alt="arch"/>

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/knowledge-base-chatbot.git
   cd knowledge-base-chatbot
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables by creating a `.env` file in the root directory:

   ```bash
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### 1. Build the Knowledge Base

Run the following command to build the knowledge base from your documents:

```bash
python build_knowledge_base.py 
```

### 2. Query the Knowledge Base (Command Line)

You can query the knowledge base using:

```bash
python query.py 
```

### 3. Launch the Gradio Chatbot Interface

To launch the chatbot web interface:

```bash
python gradio_query.py
```

This will open a web interface where you can interact with the system by asking questions.

