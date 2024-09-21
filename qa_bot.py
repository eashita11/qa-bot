from pinecone import Pinecone, ServerlessSpec, PineconeApiException
import openai
import streamlit as st
from PyPDF2 import PdfReader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Pinecone and OpenAI API setup with your keys

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY

index_name1 = "qa-bot-index"


# List all indexes to see if 'qa-bot-index' exists
indexes = pc.list_indexes()
print("Available indexes:", indexes)


try:
    pc.create_index(
        name=index_name1,
        dimension=1536,  # The model dimensions
        metric="cosine",  # The model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Index '{index_name1}' created successfully.")
    print("New Index: ", indexes)

except PineconeApiException as e:
    # Handle the case where the index already exists
    if "ALREADY_EXISTS" in str(e):
        print(f"Index '{index_name1}' already exists. Skipping creation.")
    else:
        # If it's another error, raise it
        raise e
    

# Connect to the Pinecone index
index = pc.Index(index_name1)

# Streamlit interface for document upload and question answering
st.title("QA Bot with Document Upload")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to split document into chunks
def chunk_text(text, max_tokens=500):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i:i+max_tokens])
        chunks.append(chunk)
    
    return chunks

# Function to embed document text in chunks
def embed_document(text):
    response = openai.Embedding.create(input=[text], engine='text-embedding-ada-002')
    return response['data'][0]['embedding']

# Store embeddings in Pinecone (for each chunk)
def store_embeddings_in_pinecone(chunks, doc_id):
    vectors = []
    
    for i, chunk in enumerate(chunks):
        embedding = embed_document(chunk)
        vectors.append((f"{doc_id}_chunk_{i}", embedding, {"text": chunk}))
    
    index.upsert(vectors)

if uploaded_file is not None:
    document_text = extract_text_from_pdf(uploaded_file)
    doc_id = uploaded_file.name

    # Split the document into smaller chunks
    document_chunks = chunk_text(document_text)

    # Store the embeddings for the chunks in Pinecone
    store_embeddings_in_pinecone(document_chunks, doc_id)
    st.write("Document uploaded and embeddings stored successfully!")

    query = st.text_input("Ask a question")
    
    if query:
        # Embed the query
        query_embedding = embed_document(query)

        # Search the Pinecone index for the most relevant document chunk
        results = index.query(
            vector=query_embedding,  
            top_k=5,
            include_metadata=True
        )
        
        if results and 'matches' in results:
            # Get the best match
            best_match = results['matches'][0]['metadata']['text']
            
            # Using GPT-3.5-turbo or GPT-4 to generate a detailed answer based on the best-matched chunk
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Answer this question based on the following document snippet: {query}\n\nDocument Snippet: {best_match}"}
                ],
                max_tokens=300,
                temperature=0.5
            )
            
            answer = response['choices'][0]['message']['content'].strip()
            st.write(f"Answer: {answer}")
        else:
            st.write("No relevant section found in the document.")