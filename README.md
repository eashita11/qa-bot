# RAG Model for a QA Bot with Document Upload Feature
This project implements a Question-Answering (QA) bot that allows users to upload documents (PDF) and ask questions about the content of the document. The bot uses Pinecone for vector indexing and OpenAI to generate detailed answers. The entire pipeline is containerized using Docker for easy deployment.

## Features
* Upload PDF documents
* Retrieve contextually relevant answers based on the document content
* Handle multiple queries efficiently
* Supports detailed and contextual answers
## Technology Stack
* Python
* OpenAI API (for generating responses)
* Pinecone (for vector indexing and search)
* Streamlit (for the frontend interface)
* Docker (for containerization)
## Installation and Setup
### Clone the repository
```ruby
git clone https://github.com/eashita/qa-bot-project.git
cd qa-bot-project
```
### Environment Variables
1. Create a .env file in the root directory of the project.
2. Add your OpenAI and Pinecone API keys to the .env file as follows:
```ruby
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
```
### Build and Run the Docker Container
1. Build the Docker image:
```ruby
docker build -t my-qa-bot .
```
2. Run the Docker container:
```ruby
docker run --env-file .env -p 8501:8501 my-qa-bot
```
3. Access the Streamlit app: Open your browser and go to:
```ruby
http://localhost:8501
```
### Usage
1. Upload a PDF: The application will prompt you to upload a PDF document. The content of the PDF will be processed and indexed for querying.
2. Ask a Question: Once the document is uploaded, you can input a query, and the QA bot will provide a detailed, contextually relevant answer based on the document.
3. Multiple Queries: You can ask multiple questions, and the bot will answer based on the document's context.

## Example Interactions
### Example 1
![](/Users/anil/Desktop/Example1.png)
### Example 2
### Example 3

## Requirements
To run the bot locally, you need the following:
* Docker
* Python 3.9+
* OpenAI API Key
* Pinecone API Key

## Directory Structure
```
qa-bot-project/
├── qa_bot.py            # Main Python file for the bot logic
├── requirements.txt     # Python dependencies
├── Dockerfile           # Dockerfile for containerization
├── .env                 # Environment variables (API Keys)
├── README.md            # Project documentation
```
