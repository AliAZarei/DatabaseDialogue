import argparse
import os
import shutil
import logging
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.schema.document import Document
from langchain_chroma import Chroma  # Updated import for Chroma

# Constants
CHROMA_PATH = "database"
DATA_PATH = "source_docs"

# Configure logging for production
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def get_embedding_function():
    """Return the embedding function using Ollama embeddings."""
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return embeddings
    except Exception as e:
        logging.error(f"Error initializing embeddings: {e}")
        raise

def main():
    """Main function to reset and update the Chroma database."""
    parser = argparse.ArgumentParser(description="Manage Chroma database.")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        logging.info("Clearing the database as per --reset flag.")
        clear_database()

    try:
        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(chunks)
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        raise

def load_documents():
    """Load documents from the specified directory."""
    if not os.path.exists(DATA_PATH):
        logging.error(f"Source directory '{DATA_PATH}' does not exist.")
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    
    logging.info(f"Loading documents from {DATA_PATH}.")
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    
    if not documents:
        logging.warning("No documents were found in the source directory.")
    
    return documents

def split_documents(documents):
    """Split the documents into chunks for processing."""
    if not documents:
        logging.warning("No documents to split.")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    logging.info(f"Splitting {len(documents)} documents into chunks.")
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks):
    """Add new document chunks to the Chroma database."""
    if not chunks:
        logging.warning("No chunks to add to the database.")
        return
    
    # Load or create a Chroma database
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    
    logging.info(f"🔍 Existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        logging.info(f"➕ Adding {len(new_chunks)} new documents to the database.")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        logging.info("🆗 No new documents to add.")

def calculate_chunk_ids(chunks):
    """Generate unique IDs for each document chunk."""
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If page information is missing, handle it appropriately
        if page is None:
            logging.warning("Chunk is missing page metadata.")
            continue

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    """Clear the existing Chroma database."""
    if os.path.exists(CHROMA_PATH):
        logging.info(f"Removing existing database at {CHROMA_PATH}.")
        shutil.rmtree(CHROMA_PATH)
    else:
        logging.info("No database found to clear.")

if __name__ == "__main__":
    main()
