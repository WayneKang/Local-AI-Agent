from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document # This import is necessary for creating documents
import os
import pandas as pd

df = pd.read_csv("realistic_restaurant_reviews.csv")  # Load the pizza reviews dataset
embeddings = OllamaEmbeddings(model="mxbai-embed-large")  # Create an instance of OllamaEmbeddings with the specified model
db_location = "./chroma_langchain_db"  # Specify the location for the Chroma database
add_documents = not os.path.exists(db_location)  # Check if the database already exists

if add_documents:
    documents = []
    ids = []
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],  # Use the review text as the content of the document
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id = str(i)  # Store the review ID in the metadata
        )
        ids.append(str(i))  # Collect the IDs for later use
        documents.append(document)  # Append the document to the list

vector_store = Chroma(
    collection_name = "restrant_reviews",  # Use the list of documents created from the reviews
    embedding_function = embeddings,  # Use the Ollama embeddings for vectorization
    persist_directory = db_location,  # Specify the directory to persist the database
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)  # Add the documents to the vector store with their IDs

retriver = vector_store.as_retriever(
    search_kwargs={"k": 5}  # Set the number of documents to retrieve for each query    
)  # Create a retriever from the vector store