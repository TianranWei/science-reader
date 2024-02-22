from pydantic.v1 import BaseModel
from embedding import FolderIndex
from langchain.chat_models.base import BaseChatModel
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from typing import List
from langchain.prompts import PromptTemplate


class Answers(BaseModel):
    answer: str
    sources: List[Document]


def get_answers_and_sources(
    query: str,
    folder_index: FolderIndex,
    llm: BaseChatModel,
    prompt: str
) -> Answers:
    print("Using the template: ", prompt)

    chain = load_qa_with_sources_chain(
        llm=llm,
        chain_type="stuff",
        prompt=prompt,
    )

    relevant_docs = folder_index.index.similarity_search(query, k=5)
    result = chain(
        {"input_documents": relevant_docs, "question": query}, return_only_outputs=True
    )
    sources = relevant_docs

    answer = result["output_text"].split("SOURCES: ")[0]
    print("answer:", answer)
    print("sources:", sources)

    return Answers(answer=answer, sources=sources)


def get_sources(answer: str, folder_index: FolderIndex) -> List[Document]:
    """Retrieves the docs that were used to answer the question the generated answer."""

    source_keys = [s for s in answer.split("SOURCES: ")[-1].split(", ")]

    source_docs = []
    for file in folder_index.files:
        for doc in file.docs:
            if doc.metadata["source"] in source_keys:
                source_docs.append(doc)
    return source_docs
