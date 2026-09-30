import os
from pathlib import Path
from typing import Generator, List, Tuple, Optional, Any
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv(override=True)

DEFAULT_MODEL = "qwen/qwen3.8-27b"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
DEFAULT_RETRIEVAL_K = 5

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

SYSTEM_PROMPT = """You are a knowledgeable, professional, and friendly AI assistant representing Insurellm, an enterprise insurance and InsurTech company.
You are assisting users with information regarding Insurellm's products (such as Homellm, Bizllm, Carllm, Healthllm, etc.), corporate policies, contracts, and team members.

Guidelines:
1. Base your answers strictly on the provided Context below whenever applicable.
2. If the context does not contain enough information to answer accurately, explicitly state that you don't have that information rather than fabricating details.
3. Keep your tone professional, clear, helpful, and corporate.
4. Format your response cleanly using markdown (bullet points, bold highlights, tables where relevant).

Context:
{context}
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
default_llm = ChatGroq(temperature=0, model_name=DEFAULT_MODEL)


def get_llm(temperature: float = 0.0, model_name: str = DEFAULT_MODEL) -> ChatGroq:
    """Return a ChatGroq instance with specified temperature and model."""
    return ChatGroq(temperature=temperature, model_name=model_name)


def fetch_context(question: str, k: int = DEFAULT_RETRIEVAL_K) -> List[Document]:
    """
    Retrieve relevant context documents for a question using vector similarity search.
    """
    try:
        return vectorstore.similarity_search(question, k=k)
    except Exception as e:
        print(f"Error fetching context: {e}")
        return []


def normalize_history(history: Optional[List[Any]]) -> List[BaseMessage]:
    """
    Safely convert various Gradio history formats (list of dicts, list of tuples, etc.)
    into LangChain BaseMessage objects.
    """
    messages: List[BaseMessage] = []
    if not history:
        return messages

    for item in history:
        if isinstance(item, dict):
            role = item.get("role", "")
            content = item.get("content", "")
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=str(content)))
            elif role == "assistant":
                messages.append(AIMessage(content=str(content)))
            elif role == "system":
                messages.append(SystemMessage(content=str(content)))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, bot_msg = item
            if user_msg:
                messages.append(HumanMessage(content=str(user_msg)))
            if bot_msg:
                messages.append(AIMessage(content=str(bot_msg)))

    return messages


def combined_question(question: str, history: Optional[List[Any]] = None) -> str:
    """
    Combine user messages from history with the current question to maintain context for retrieval.
    """
    if not history:
        return question

    user_queries: List[str] = []
    for item in history:
        if isinstance(item, dict) and item.get("role") == "user":
            user_queries.append(str(item.get("content", "")))
        elif isinstance(item, (list, tuple)) and len(item) == 2 and item[0]:
            user_queries.append(str(item[0]))

    if not user_queries:
        return question

    # Return the last few queries plus current question for focused retrieval
    prior = "\n".join(user_queries[-2:])
    return f"{prior}\n{question}"


def answer_question(
    question: str,
    history: Optional[List[Any]] = None,
    k: int = DEFAULT_RETRIEVAL_K,
    temperature: float = 0.0,
    model_name: str = DEFAULT_MODEL,
    system_prompt_template: str = SYSTEM_PROMPT
) -> Tuple[str, List[Document]]:
    """
    Answer the given question with RAG; returns (answer_text, docs).
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = system_prompt_template.format(context=context)

    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
    messages.extend(normalize_history(history))
    messages.append(HumanMessage(content=question))

    llm = get_llm(temperature=temperature, model_name=model_name)
    response = llm.invoke(messages)
    return str(response.content), docs


def stream_answer_question(
    question: str,
    history: Optional[List[Any]] = None,
    k: int = DEFAULT_RETRIEVAL_K,
    temperature: float = 0.0,
    model_name: str = DEFAULT_MODEL,
    system_prompt_template: str = SYSTEM_PROMPT
) -> Generator[Tuple[str, List[Document]], None, None]:
    """
    Stream RAG response token by token; yields (partial_answer_text, docs).
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = system_prompt_template.format(context=context)

    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
    messages.extend(normalize_history(history))
    messages.append(HumanMessage(content=question))

    llm = get_llm(temperature=temperature, model_name=model_name)
    partial_text = ""
    for chunk in llm.stream(messages):
        delta = chunk.content
        if isinstance(delta, str):
            partial_text += delta
        elif isinstance(delta, list):
            for part in delta:
                if isinstance(part, dict) and "text" in part:
                    partial_text += part["text"]
                elif isinstance(part, str):
                    partial_text += part
        yield partial_text, docs

