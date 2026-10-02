from pathlib import Path
from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential, stop_after_attempt
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv(override=True)

llm_model = "groq/qwen/qwen3.8-27b"

# Support preprocessed_db or vector_db if present
db_path = Path(__file__).parent.parent / "preprocessed_db"
if not db_path.exists():
    db_path = Path(__file__).parent.parent / "vector_db"
DB_NAME = str(db_path)

KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
SUMMARIES_PATH = Path(__file__).parent.parent / "summaries"

embedding_model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

wait = wait_exponential(multiplier=1, min=2, max=10)
stop = stop_after_attempt(3)

chroma = PersistentClient(path=DB_NAME)
existing_collections = [c.name for c in chroma.list_collections()]
collection_name = "docs" if "docs" in existing_collections else ("langchain" if "langchain" in existing_collections else "docs")
collection = chroma.get_or_create_collection(collection_name)

RETRIEVAL_K = 20
FINAL_K = 10

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""


class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


@retry(wait=wait, stop=stop, reraise=True)
def _rerank_call(question, chunks):
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=llm_model, messages=messages, response_format=RankOrder)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    valid_order = [i - 1 for i in order if isinstance(i, int) and 0 <= i - 1 < len(chunks)]
    remaining = [idx for idx in range(len(chunks)) if idx not in valid_order]
    return [chunks[idx] for idx in valid_order + remaining]


def rerank(question, chunks):
    if not chunks or len(chunks) <= 1:
        return chunks
    try:
        return _rerank_call(question, chunks)
    except Exception as e:
        print(f"Rerank failed: {e}, using original order")
        return chunks


def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {chunk.metadata.get('source', '')}:\n{chunk.page_content}" for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


@retry(wait=wait, stop=stop, reraise=True)
def _rewrite_query_call(question, history=None):
    if history is None:
        history = []

    system_message = """You are in a conversation with a user.
You are about to look up information in a Knowledge Base to answer the user's question.
Since the conversation is contextual, understand the meaning of the user question and add details based on the history.
Condense everything in a single contextually-rich VERY short and specific question, most likely to surface content.

EXAMPLE:
user: Who is the founder? -> Query: who is the founder?
assistant: The founder is FooBar
user: What role covers? -> Query: What role FooBar covers?

IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else."""

    user_message = f"History:\n{history}\n\nCurrent question:\n{question}"

    response = completion(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content.strip()


def rewrite_query(question, history=None):
    try:
        return _rewrite_query_call(question, history)
    except Exception as e:
        print(f"Rewrite query failed: {e}, using original question")
        return question


def merge_chunks(chunks, reranked):
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


def fetch_context_unranked(question):
    query = embeddings.embed_query(question)
    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)
    chunks = []
    if results and "documents" in results and results["documents"] and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if ("metadatas" in results and results["metadatas"]) else [{}] * len(docs)
        for doc, meta in zip(docs, metas):
            chunks.append(Result(page_content=doc, metadata=meta or {}))
    return chunks


def fetch_context(original_question, history=None, k: int = FINAL_K):
    rewritten_question = rewrite_query(original_question, history)
    print(rewritten_question)
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    chunks = merge_chunks(chunks1, chunks2)
    reranked = rerank(original_question, chunks)
    return reranked[:k]


@retry(wait=wait)
def answer_question(question: str, history: list[dict] = [], k: int = FINAL_K) -> tuple[str, list]:
    """
    Answer a question using RAG and return the answer and the retrieved context
    """
    chunks = fetch_context(question, history, k=k)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=llm_model, messages=messages)
    return response.choices[0].message.content, chunks
