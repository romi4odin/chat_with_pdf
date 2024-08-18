import boto3
import streamlit as st
import os
import uuid

##s3 client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

#Bedrock
from langchain_community.embeddings import BedrockEmbeddings

#Text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Pdf Loader
from langchain_community.document_loaders import PyPDFLoader

# import FAISS

from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name="bedrock-runtime")

bedrock_embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

folder_path="/tmp/"

def get_unique_id():
    return str(uuid.uuid4())

def create_vector_store(request_id, documents):
    vectorestore_faiss = FAISS.from_documents(documents, bedrock_embedding)
    file_name = f"{request_id}.bin"
    folder_path = "/tmp/"
    vectorestore_faiss.save_local(index_name=file_name , folder_path=folder_path)

    ## upload to s3
    s3_client.upload_file(Filename = folder_path + "/" + file_name + ".faiss", Bucket = BUCKET_NAME, Key = "my_faiss.faiss")
    s3_client.upload_file(Filename = folder_path + "/" + file_name + ".pkl", Bucket = BUCKET_NAME, Key = "my_faiss.pkl")

    return True

#load index
def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}myfaiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}myfaiss.pkl")

def main():
    st.header("This is Client Site for chat with PDF demo using Bedrock")

    load_index()

    dir_list = os.listdir(folder_path)
    st.write(f"Files and Directories in {folder_path}")
    st.write(dir_list)

if __name__=="__main__":
    main()