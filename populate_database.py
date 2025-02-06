# import argparse
# import os
# import shutil
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.schema.document import Document
# from get_embedding_function import get_embedding_function

# CHROMA_PATH = "chroma"
# # DATA_PATH = "/Users/dxchmxww/Desktop/rag-tutorial-v2/data/KMUTTRegulation.txt"  # Use a string, not a tuple
# DATA_PATH = "/Users/dxchmxww/Desktop/rag-tutorial-v2/data/monopoly.pdf" 
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--reset", action="store_true", help="Reset the database.")
#     args = parser.parse_args()
    
#     if args.reset:
#         print("✨ Clearing Database")
#         clear_database()

#     documents = load_documents()
#     print(documents[0])
#     chunks = split_documents(documents)
#     add_to_chroma(chunks)

# def load_documents():
#     document_loader = PyPDFLoader(DATA_PATH)  # Use PyPDFLoader for a single PDF
#     return document_loader.load()

# def split_documents(documents: list[Document]):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=80,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     return text_splitter.split_documents(documents)

# def add_to_chroma(chunks: list[Document]):
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    
#     chunks_with_ids = calculate_chunk_ids(chunks)
#     existing_items = db.get(include=[])  
#     existing_ids = set(existing_items["ids"])
    
#     print(f"Number of existing documents in DB: {len(existing_ids)}")
    
#     new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    
#     if new_chunks:
#         print(f"👉 Adding new documents: {len(new_chunks)}")
#         new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
#         db.add_documents(new_chunks, ids=new_chunk_ids)
#         db.persist()
#     else:
#         print("✅ No new documents to add")

# def calculate_chunk_ids(chunks):
#     last_page_id = None
#     current_chunk_index = 0

#     for chunk in chunks:
#         source = chunk.metadata.get("source")
#         page = chunk.metadata.get("page")
#         current_page_id = f"{source}:{page}"

#         if current_page_id == last_page_id:
#             current_chunk_index += 1
#         else:
#             current_chunk_index = 0

#         chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
#         last_page_id = current_page_id

#     return chunks

# def clear_database():
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)

# if __name__ == "__main__":
#     main()

import argparse
import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
# DATA_PATH = "data"
DATA_PATH = "/Users/dxchmxww/Desktop/rag-tutorial-v2/data/RG.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    if not documents:
        print("❌ No documents were loaded. Exiting.")
        return

    print(f"✅ Loaded {len(documents)} document(s).")
    print("🔹 First document preview:\n", documents[0])

    chunks = split_documents(documents)
    add_to_chroma(chunks)
    
def load_documents():
    """Load and parse the .txt file."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"🚨 File not found: {DATA_PATH}")

    try:
        # Try TIS-620 encoding for Thai
        document_loader = TextLoader(DATA_PATH, encoding="tis-620")
        return document_loader.load()
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load text file: {e}")



def split_documents(documents: list[Document]):
    """Split documents into smaller text chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    """Store document chunks in Chroma with unique IDs."""
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])  
    existing_ids = set(existing_items.get("ids", []))  # Handle empty DB case
    
    print(f"📊 Existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    
    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    """Generate unique IDs for document chunks."""
    source = os.path.basename(DATA_PATH)  # Use filename as source
    current_chunk_index = 0

    for chunk in chunks:
        chunk.metadata["id"] = f"{source}:{current_chunk_index}"
        current_chunk_index += 1

    return chunks

def clear_database():
    """Clear the Chroma database."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("🧹 Database cleared!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
