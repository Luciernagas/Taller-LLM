from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.tools import tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
import pinecone
import os


@tool("SayHello", return_direct=True)
def say_hello(name: str) -> str:
    """Answer when someone says hello"""
    return f"Hello {name}! My name is Sainapsis"


def main_chat():
    llm = ChatOpenAI(temperature=0, openai_api_key="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")
    tools = [say_hello]
    agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
    print(agent.run("Hello! My name is Luisa"))


def main_pinecone():
    pinecone.init(api_key="48139c27-fa66-49f9-a37a-c1fb75ad573e", environment="gcp-starter")
    embeddings = OpenAIEmbeddings()
    texts = ["economia.txt", "ingenieria-civil.txt", "ingenieria-sistemas.txt", "ingenieria-electrica.txt",
             "ingenieria-industrial.txt"]

    for text_file in texts:
        with open(text_file, "r") as text:
            data = Pinecone.from_texts(texts=text.readlines(), embedding=embeddings, index_name='db')


def buscar():
    pinecone.init(api_key="48139c27-fa66-49f9-a37a-c1fb75ad573e", environment="gcp-starter")
    embeddings = OpenAIEmbeddings()

    docsearch = Pinecone.from_existing_index("db", embeddings)
    query = "Cuantos años de acreditación tiene ingeniería de industrial?"
    docs = docsearch.similarity_search(query)
    print(docs)


if __name__ == "__main__":
    main_chat()
    main_pinecone()
    buscar()
