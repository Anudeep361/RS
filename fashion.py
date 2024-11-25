import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import tiktoken
import os
import openai
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI
import warnings
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

# Load the dataset
file_path = 'Fashion_products_catalog.csv'
df = pd.read_csv(file_path)

# Display basic info about the dataset
print("Dataset Info:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

df['PrimaryColor'].fillna('Unknown', inplace=True)

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# API Key Setup
api_key = 'OPENAI_API_KEY'
os.environ['OPENAI_API_KEY'] = api_key

def recommend(query):
    # Load data from CSV file
    loader = CSVLoader(file_path="fashion_updated.csv")
    data = loader.load()

def recommend(query):
    # Load data from CSV file
    loader = CSVLoader(file_path="fashion_updated.csv")
    data = loader.load()

    # Split text into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    # Embeddings Model using OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Vector Database to store and retrieve documents
    docsearch = Chroma.from_documents(texts, embeddings)

    # Create a Prompt Template for recommendations
    template = """
    You are a recommendation system for a fashion store that assists users in finding products that match their needs. For each query, suggest five products from the store with the following details:

    Generate a product recommendation response for a fashion product in the following format:
    Product Name:
    - Description:
    - Price (INR):
    - Gender:
    - Number of Images:
    - Primary Color:
    - Brand:

    {context}

    Question: {question}
    Your response:
    """
    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"])

    chain_type_kwargs = {"prompt": PROMPT}

    # Define the language model to use for retrieval-based QA
    llm = ChatOpenAI(model_name='gpt-4', temperature=0)  # Changed to GPT-4

    # Create the RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

    # Run the QA system with the query and retrieve result
    result = qa({'query': query})

    # Return the generated product recommendations
    return result['result']

import time
import openai

def recommend_with_backoff(query, max_retries=5):
    for i in range(max_retries):
        try:
            return recommend(query)  # Call your recommendation function
        except openai.error.RateLimitError:
            if i == max_retries - 1:
                raise
            wait_time = 2 ** i  # Exponential backoff
            print(f"Rate limit hit. Retrying in {wait_time} seconds.")
            time.sleep(wait_time)
