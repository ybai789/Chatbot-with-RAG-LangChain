import os
import logging
from typing import List
import argparse
from dotenv import load_dotenv, find_dotenv
import shutil
from tqdm import tqdm
import concurrent.futures
import time
import random

from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(find_dotenv())

class KnowledgeBase:
    def __init__(self, data_path_md: str, data_path_pdf: str, chroma_path: str, chunk_size: int = 1000, chunk_overlap: int = 500):
        self.data_path_md = data_path_md
        self.data_path_pdf = data_path_pdf
        self.chroma_path = chroma_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("No OpenAI API key found in environment variables")

    def generate_knowledge_base(self, clean: bool = False):
        """Generate the knowledge base from documents."""
        logging.info("Loading documents...")
        documents = self.load_documents()
        logging.info(f"Loaded {len(documents)} documents.")
        logging.info("Splitting text...")
        chunks = self.split_text(documents)
        logging.info("Saving to Chroma...")
        self.save_to_chroma(chunks, clean)

    def load_documents(self) -> List[Document]:
        """Load documents from the data path (both markdown and pdf)."""
        documents = []

        # Load markdown and PDF files
        md_files = [f for f in os.listdir(self.data_path_md) if f.endswith(".md")]
        pdf_files = [f for f in os.listdir(self.data_path_pdf) if f.endswith(".pdf")]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Load Markdown files
            future_to_md_file = {executor.submit(self.load_single_document, os.path.join(self.data_path_md, file), 'md'): file for file in md_files}
            for future in tqdm(concurrent.futures.as_completed(future_to_md_file), total=len(md_files), desc="Loading markdown documents"):
                file = future_to_md_file[future]
                try:
                    documents.extend(future.result())
                except Exception as e:
                    logging.error(f"Error loading markdown file {file}: {str(e)}")

            # Load PDF files
            future_to_pdf_file = {executor.submit(self.load_single_document, os.path.join(self.data_path_pdf, file), 'pdf'): file for file in pdf_files}
            for future in tqdm(concurrent.futures.as_completed(future_to_pdf_file), total=len(pdf_files), desc="Loading PDF documents"):
                file = future_to_pdf_file[future]
                try:
                    documents.extend(future.result())
                except Exception as e:
                    logging.error(f"Error loading PDF file {file}: {str(e)}")

        return documents

    def load_single_document(self, file_path: str, file_type: str) -> List[Document]:
        """Load a single document based on its file type."""
        if file_type == 'md':
            loader = UnstructuredMarkdownLoader(file_path)
        elif file_type == 'pdf':
            loader = PyPDFLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        return loader.load()

    def split_text(self, documents: List[Document]) -> List[Document]:
        """Split the documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True
        )

        chunks = []
        for doc in tqdm(documents, desc="Splitting documents"):
            chunks.extend(text_splitter.split_documents([doc]))

        logging.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def save_to_chroma(self, chunks: List[Document], clean: bool = False):
        """Save the chunks to Chroma database with batch processing and retry logic."""
        if clean and os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)

        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

        batch_size = 100  # Define batch size to reduce the number of API calls
        retries = 5  # Max retries for API rate limit errors

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            attempt = 0
            success = False

            while not success and attempt < retries:
                try:
                    with tqdm(total=len(batch), desc="Saving batch to Chroma") as pbar:
                        db = Chroma.from_documents(
                            batch,
                            embeddings,
                            persist_directory=self.chroma_path
                        )
                        db.persist()
                        pbar.update(len(batch))
                    success = True
                except Exception as e:
                    logging.error(f"Error saving batch: {str(e)}")
                    attempt += 1
                    wait_time = 2 ** attempt + random.uniform(0, 1)  # Exponential backoff with jitter
                    logging.info(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)

        logging.info(f"Saved {len(chunks)} chunks to knowledge base in {self.chroma_path}.")

def main():
    parser = argparse.ArgumentParser(description="Generate knowledge base from markdown and PDF documents")
    parser.add_argument("--clean", action="store_true", help="Clean existing knowledge base before saving")
    args = parser.parse_args()

    knowledge_base = KnowledgeBase(
        data_path_md="data/md",  # Path to markdown documents
        data_path_pdf="data/pdf",  # Path to PDF documents
        chroma_path="chroma"
    )
    knowledge_base.generate_knowledge_base(clean=args.clean)

if __name__ == "__main__":
    main()

