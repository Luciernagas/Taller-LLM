import os

import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone


def main():
    pinecone.init(api_key="48139c27-fa66-49f9-a37a-c1fb75ad573e", environment="gcp-starter")
    embeddings = OpenAIEmbeddings()
    text = open("economia.txt", "r")
    #print(text.read())
    pinecone.create_index("db", dimension=1536, metric="cosine")
    data = Pinecone.from_texts(texts=text.readlines(), embedding=embeddings, index_name='db')
    text = open("ingenieria-civil.txt", "r")
    data = Pinecone.from_texts(texts=text.readlines(), embedding=embeddings, index_name='db')
    text = open("ingenieria-sistemas.txt", "r")
    data = Pinecone.from_texts(texts=text.readlines(), embedding=embeddings, index_name='db')
    text = open("ingenieria-electrica.txt", "r")
    data = Pinecone.from_texts(texts=text.readlines(), embedding=embeddings, index_name='db')
    text = open("ingenieria-industrial.txt", "r")
    data = Pinecone.from_texts(texts=text.readlines(), embedding=embeddings, index_name='db')

def buscar():
    pinecone.init(api_key="48139c27-fa66-49f9-a37a-c1fb75ad573e", environment="gcp-starter")
    embeddings = OpenAIEmbeddings()
    # if you already have an index, you can load it like this
    docsearch = Pinecone.from_existing_index("db", embeddings)
    query = "Cuantos años de acreditación tiene ingeniería de industrial?"
    docs = docsearch.similarity_search(query)
    print(docs)
