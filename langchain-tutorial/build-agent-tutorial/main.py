from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

# load environment variables from .env file
load_dotenv()

# sample document URLs
urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

# chunk and create vectorstore
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc_splits = text_splitter.split_documents(docs_list)
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    name="retrieve_blog_posts",
    description="Search and return information about Lilian Weng blog posts.",
)

response_model = init_chat_model(model="gpt-4o-mini", temperature=0)


def generate_query_or_respond(state: MessagesState):
    """Generate a response, appending it to the existing messages list."""
    memory_messages = state["messages"]
    retriever = response_model.bind_tools([retriever_tool])
    response = retriever.invoke(memory_messages)
    return {"messages": memory_messages + [response]}


if __name__ == "__main__":
    input = {
        "messages": [
            {
                "role": "user",
                "content": "hello! how are you?",
            }
        ]
    }
    print(generate_query_or_respond(input)["messages"][-1].pretty_print())
