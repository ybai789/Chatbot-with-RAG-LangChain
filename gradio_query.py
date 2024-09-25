import gradio as gr
import os
import sys
from dotenv import load_dotenv, find_dotenv
import argparse

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
Answer this question based on the context above:
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
    
    return [documents[i] for i in top_indices], similarities[top_indices]

def chatbot_response(query_text):
    try:
        # Prepare the database
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the database for relevant chunks
        initial_results = db.similarity_search(query_text, k=20)

        # Apply reranking
        reranked_results, relevance_scores = rerank_documents(query_text, initial_results)

        if not reranked_results:
            return "Sorry, I couldn't find any relevant information.", "No sources available."

        # Create the prompt for the chatbot
        context = "\n\n---\n\n".join([doc.page_content[:300] for doc in reranked_results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context, question=query_text)

        # Use the LLM to answer the question
        model = ChatOpenAI()
        response = model.invoke([HumanMessage(content=prompt)])
        response_text = response.content

        # Prepare sources
        sources_info = []
        unique_sources = set()
        for doc, score in zip(reranked_results, relevance_scores):
            source = format_source_path(doc.metadata.get("source", "Unknown"))
            if source not in unique_sources:
                unique_sources.add(source)
                relevant_excerpt = extract_relevant_excerpt(doc.page_content, query_text)
                sources_info.append(f"- {source} (Relevance: {score:.2f})\n  Excerpt: {relevant_excerpt[:200]}...")
        
        sources_text = "\n".join(sources_info)
        return response_text, sources_text

    except Exception as e:
        return f"An error occurred: {str(e)}", "No sources available."

# Gradio callback function
def gradio_chat_interface(query_text):
    answer, sources = chatbot_response(query_text)
    return f"Answer: {answer}\n\nSources:\n{sources}"

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Knowledge Base Chatbot")
    query = gr.Textbox(label="Enter your question", placeholder="What is Dengue Fever?")
    output = gr.Textbox(label="Response", interactive=False)
    
    # Button for submitting questions
    submit_btn = gr.Button("Submit")
    
    # Define interaction logic when the button is clicked
    submit_btn.click(fn=gradio_chat_interface, inputs=query, outputs=output)

# Launch Gradio interface
if __name__ == "__main__":
    demo.launch()