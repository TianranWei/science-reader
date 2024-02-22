import streamlit as st
from langchain.llms import OpenAI
from pdf_parser import File, read_file
from embedding import embed_files
from langchain_community.chat_models import ChatOpenAI
from rag import get_answers_and_sources
from chunking import chunk_file
from streamlit.logger import get_logger
from prompt import PROMPT_DOCUMENT, PROMPT_GENERATIVE


MODEL_LIST = ["gpt-3.5-turbo", "gpt-4"]
EMBEDDING = "openai"
VECTOR_STORE = "faiss"
CHUNK_SIZE = 250
USER_GENERATED_ANSWER = False
BIGGER_CHUNKING_SIZE = False
PROMPT = PROMPT_DOCUMENT

st.title('Science Reader 🥸')
# bootstrap_caching()


openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
if not openai_api_key:
    st.warning(
        "Enter your OpenAI API key in the sidebar ：)"
    )


with st.expander("Advanced Options"):
    USER_GENERATED_ANSWER = st.checkbox(
        "Let the model generate the answer if no related information found in the document")
    BIGGER_CHUNKING_SIZE = st.checkbox(
        "Use bigger chunking size for more contextual information (recommended for longer documents)")
    USE_ = st.checkbox(
        "Use BERT Sentence Transformers instead of OpenAI for embedding(recommended for shorter documents and expected answers)")
print('use_generated_answer: ', USER_GENERATED_ANSWER)
if USER_GENERATED_ANSWER:
    PROMPT = PROMPT_GENERATIVE


if BIGGER_CHUNKING_SIZE:
    CHUNK_SIZE = 450


uploaded_file = st.file_uploader(
    "Upload a pdf file to build your own brain!",
    type=["pdf"],
    help="Scanned documents are not supported yet!",
)


if not uploaded_file:
    st.stop()

try:
    file = read_file(uploaded_file)
except Exception as e:
    print(e)

# chunck file
chunked_file = chunk_file(file, chunk_size=400, chunk_overlap=0)

logger = get_logger(__name__)


def is_file_valid(file: File) -> bool:
    if (
        len(file.docs) == 0
        or "".join([doc.page_content for doc in file.docs]).strip() == ""
    ):
        st.error("Cannot read document! Make sure the document has selectable text")
        logger.error("Cannot read document")
        return False
    return True


if not is_file_valid(file):
    st.stop()

# embedding the documents
model: str = st.selectbox("Model", options=MODEL_LIST)  # type: ignore


with st.spinner("Indexing document... This may take a while⏳"):
    folder_index = embed_files(
        files=[chunked_file],
        embedding=EMBEDDING if model != "debug" else "debug",
        vector_store=VECTOR_STORE if model != "debug" else "debug",
        openai_api_key=openai_api_key,
    )


def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    st.info(llm(input_text))


with st.form('my_form'):
    text = st.text_area(
        'Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')

if submitted:

    # Output Columns
    answer_col, sources_col = st.columns(2)

    llm = ChatOpenAI(model=model, openai_api_key=openai_api_key, temperature=0)
    result = get_answers_and_sources(
        folder_index=folder_index,
        query=text,
        llm=llm,
        prompt=PROMPT
    )

    with answer_col:
        st.markdown("#### Answer")
        st.markdown(result.answer)

    with sources_col:
        st.markdown("#### Sources")
        for source in result.sources:
            st.markdown(source.page_content)
            source_note = f"From page {source.metadata['page']}, chunk {source.metadata['chunk']}"
            st.markdown(source_note)
            st.markdown("---")
