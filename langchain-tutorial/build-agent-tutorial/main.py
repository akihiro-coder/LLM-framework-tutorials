from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState  # MessagesState: LangGraph のステート型。会話履歴(messagesリスト)を保持し、ノード関数内で state["messages"] で取得・更新する。関数は {"messages": [...]} を返すことで履歴がマージされ次のノードへ渡される。
from pydantic import BaseModel, Field

# load environment variables from .env file
load_dotenv()


class PreProcess:
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


class Node:

    def generate_query_or_respond(state: MessagesState):
        """Generate a response, appending it to the existing messages list."""
        memory_messages = state["messages"]
        response_model = init_chat_model(model="gpt-4o-mini", temperature=0)
        retriever = response_model.bind_tools([retriever_tool])
        response = retriever.invoke(memory_messages)
        return {"messages": memory_messages + [response]}


    REWRITE_PROMPT = (
        "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
        "Here is the initial question:"
        "\n ------- \n"
        "{question}"
        "\n ------- \n"
        "Formulate an improved question:"
    )



class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class Edge:

    # GRADE_PROMPT:
    #   検索で取得した文書(context)がユーザ質問(question)に関連しているかを評価するための指示。
    #   キーワードまたは意味的関連があれば「yes」、無ければ「no」の二値判定を返すようモデルに求める。
    GRADE_PROMPT = (
        "You are a grader assessing relevance of a retrieved document to a user question. \n "
        "Here is the retrieved document: \n\n {context} \n\n"
        "Here is the user question: {question} \n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )

    def grade_documents(
        state: MessagesState,
    ) -> Literal["generate_answer", "rewrite_question"]:
        """Determine whether to generate an answer or rewrite the question based on document relevance."""

        question = state["messages"][0].content
        context = state["messages"][-1].content

        prompt = GRADE_PROMPT.format(question=question, context=context)
        grader_model = init_chat_model(model="gpt-4o-mini", temperature=0)
        response = (
            grader_model.with_structured_output(GradeDocuments).invoke(
                [{"role": "user", "content": prompt}]
            )
        )
        score = response.binary_score
        if score == "yes":
            return "generate_answer"
        else:
            return "rewrite_question"



















if __name__ == "__main__":


    from langchain_core.messages import convert_to_messages

    input = {
        "messages": convert_to_messages(
            [
                {
                    "role": "user",
                    "content": "What does Lilian Weng say about types of reward hacking?",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "retrieve_blog_posts",
                            "args": {"query": "types of reward hacking"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "meow",
                    "tool_call_id": "1"
                },
            ]
        )
    }
    response = grade_documents(input) # "rewrite_question" is expected
    print(response) # rewrite_question


    input2 = {
        "messages": convert_to_messages(
            [
                {
                    "role": "user",
                    "content": "What does Lilian Weng say about types of reward hacking?",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "retrieve_blog_posts",
                            "args": {"query": "types of reward hacking"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
                    "tool_call_id": "1",
                },
            ]
        )
    }
    response2 = grade_documents(input2) # "generate_answer" is expected
    print(response2) # generate_answer
