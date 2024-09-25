import os
import sys
from dotenv import load_dotenv, find_dotenv
import argparse
from tqdm import tqdm
import requests

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment variables
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY not found in environment variables.")
    sys.exit(1)

CHROMA_PATH = "chroma"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = f"./{MODEL_NAME}"
PROMPT_TEMPLATE = """
Answer the question using only the following context:
{context}
-------------------------------------------------------------
Based on the above context, answer this question:
{question}
"""

def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}. Skipping download.")
        return

    print(f"Downloading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    model.save(MODEL_PATH)
    print(f"Model downloaded and saved to {MODEL_PATH}")

# Initialize Sentence Transformer model
try:
    download_model()
    rerank_model = SentenceTransformer(MODEL_PATH)
    print("Successfully loaded model.")
except Exception as e:
    print(f"Error occurred while loading the model: {str(e)}")
    sys.exit(1)

def format_source_path(full_path):
    return os.path.basename(full_path)

def extract_relevant_excerpt(doc_content, query):
    lower_content = doc_content.lower()
    query_keywords = query.lower().split()
    for keyword in query_keywords:
        if keyword in lower_content:
            start_pos = max(0, lower_content.find(keyword) - 100)
            end_pos = min(len(doc_content), lower_content.find(keyword) + 150)
            return doc_content[start_pos:end_pos].replace("\n", " ")
    return doc_content[:150].replace("\n", " ")

def rerank_documents(query, documents, top_k=4):
    query_embedding = rerank_model.encode([query])[0]
    doc_embeddings = rerank_model.encode([doc.page_content for doc in documents])
    
    similarities = np.dot(doc_embeddings, query_embedding)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    return [documents[i] for i in top_indices]

def chatbot_response(query_text):
    try:
        # Prepare the database
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search for relevant text chunks in the database
        initial_results = db.similarity_search(query_text, k=20)

        # Apply reranking
        reranked_results = rerank_documents(query_text, initial_results)

        if not reranked_results:
            print(f"Unable to find matching results for '{query_text}'")
            return

        # Create prompt for the chatbot
        context = "\n\n---\n\n".join([doc.page_content[:300] for doc in reranked_results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context, question=query_text)

        # Use LLM to answer the question
        model = ChatOpenAI()
        response = model.invoke([HumanMessage(content=prompt)])
        response_text = response.content

        # Format and print the answer
        print("\nQuestion:", query_text)
        print("\nAnswer:", response_text)
        print("\nSources:")

        # Format and print sources
        unique_sources = set()
        for doc in reranked_results:
            source = format_source_path(doc.metadata.get("source", "Unknown"))
            if source not in unique_sources:
                unique_sources.add(source)
                print(f"- {source}")
                relevant_excerpt = extract_relevant_excerpt(doc.page_content, query_text)
                print(f"  Excerpt: {relevant_excerpt[:200]}...")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Query the Knowledge Base")
    parser.add_argument("query", nargs="?", type=str, help="The question to ask")
    args = parser.parse_args()

    if args.query:
        chatbot_response(args.query)
    else:
        print("Welcome to the Knowledge Base. Enter your question below, or type 'quit' to exit.")
        while True:
            query = input("\nEnter your question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            elif not query.strip():
                print("Please enter a valid question.")
            else:
                chatbot_response(query)

if __name__ == "__main__":
    main()
