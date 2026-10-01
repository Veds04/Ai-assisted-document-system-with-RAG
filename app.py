import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# Streamlit Page Configuration
# ============================================================

st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Chat with your PDF")
st.caption("Ask questions and get answers based only on the uploaded document.")


# ============================================================
# Cached Resources
# ============================================================

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash"
    )


# ============================================================
# Prompt
# ============================================================

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say exactly:

"I could not find the answer in the document."

Keep your answer clear, concise, and easy to understand.

Do not return JSON, Python objects, metadata,
signatures, or any internal information."""
        ),
        (
            "human",
            """Context:

{context}

Question:

{question}
"""
        ),
    ]
)


# ============================================================
# Response Cleaning
# ============================================================

def extract_response_text(response):
    """
    Extract only the actual text from a LangChain AIMessage.

    Gemini/LangChain can return response.content as either:
    - a string
    - a list of content blocks
    """

    content = response.content

    # Normal string response
    if isinstance(content, str):
        return content.strip()

    # List of content blocks
    if isinstance(content, list):

        text_parts = []

        for block in content:

            # Example:
            # {
            #     "type": "text",
            #     "text": "actual answer",
            #     "extras": {...}
            # }
            if isinstance(block, dict):

                if block.get("type") == "text":

                    text = block.get("text", "")

                    if text:
                        text_parts.append(text)

            # Handle plain strings inside the list
            elif isinstance(block, str):

                text_parts.append(block)

        return "\n".join(text_parts).strip()

    # Fallback
    return str(content).strip()


# ============================================================
# Session State
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


if "retriever" not in st.session_state:
    st.session_state.retriever = None


if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None


# ============================================================
# Build Retriever
# ============================================================

def build_retriever_from_pdf(uploaded_file):
    """
    Read the uploaded PDF, split it into chunks,
    create embeddings, and create an in-memory Chroma vector store.
    """

    # --------------------------------------------------------
    # Save uploaded file temporarily
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)

    try:

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        loader = PyPDFLoader(str(tmp_path))

        docs = loader.load()

    finally:

        # Remove temporary PDF
        tmp_path.unlink(missing_ok=True)


    # --------------------------------------------------------
    # Check PDF
    # --------------------------------------------------------

    if not docs:

        st.error(
            "Could not extract any text from this PDF. "
            "It may be scanned or image-only."
        )

        return None


    # --------------------------------------------------------
    # Split documents
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=140
    )

    chunks = splitter.split_documents(docs)


    if not chunks:

        st.error("Could not create text chunks from this PDF.")

        return None


    # --------------------------------------------------------
    # Create Chroma vector store
    # --------------------------------------------------------

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model()
    )


    # --------------------------------------------------------
    # Create retriever
    # --------------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    return retriever


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("📄 Upload a PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )


    if uploaded_file is not None:

        file_id = f"{uploaded_file.name}-{uploaded_file.size}"


        # ----------------------------------------------------
        # Process only when a new file is uploaded
        # ----------------------------------------------------

        if file_id != st.session_state.current_file_id:

            with st.spinner(
                "Reading and indexing your PDF..."
            ):

                retriever = build_retriever_from_pdf(
                    uploaded_file
                )


            if retriever is not None:

                st.session_state.retriever = retriever

                st.session_state.current_file_id = file_id

                # Clear previous conversation
                st.session_state.messages = []

                st.success(
                    f"Indexed: {uploaded_file.name}"
                )


    # --------------------------------------------------------
    # Current document
    # --------------------------------------------------------

    if st.session_state.retriever is not None:

        current_file = (
            st.session_state.current_file_id
            .rsplit("-", 1)[0]
        )

        st.caption(
            f"Currently loaded: **{current_file}**"
        )

    else:

        st.caption(
            "No PDF loaded yet."
        )


# ============================================================
# Display Previous Messages
# ============================================================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])


# ============================================================
# Main Chat
# ============================================================

if st.session_state.retriever is None:

    st.info(
        "👈 Upload a PDF from the sidebar to start chatting."
    )

else:

    query = st.chat_input(
        "Ask a question about the PDF..."
    )


    if query:

        # ----------------------------------------------------
        # Display user question
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )


        with st.chat_message("user"):

            st.markdown(query)


        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    # ----------------------------------------
                    # Retrieve relevant chunks
                    # ----------------------------------------

                    docs = st.session_state.retriever.invoke(
                        query
                    )


                    # ----------------------------------------
                    # No documents found
                    # ----------------------------------------

                    if not docs:

                        answer = (
                            "I could not find the answer "
                            "in the document."
                        )

                    else:

                        # ------------------------------------
                        # Build context
                        # ------------------------------------

                        context = "\n\n".join(
                            doc.page_content
                            for doc in docs
                        )


                        # ------------------------------------
                        # Create prompt
                        # ------------------------------------

                        final_prompt = PROMPT.invoke(
                            {
                                "context": context,
                                "question": query
                            }
                        )


                        # ------------------------------------
                        # Call Gemini
                        # ------------------------------------

                        response = get_llm().invoke(
                            final_prompt
                        )


                        # ------------------------------------
                        # Extract clean answer
                        # ------------------------------------

                        answer = extract_response_text(
                            response
                        )


                    # ----------------------------------------
                    # Display answer
                    # ----------------------------------------

                    st.markdown(answer)


                    # ----------------------------------------
                    # Retrieved Sources
                    # ----------------------------------------

                    if docs:

                        with st.expander(
                            "📚 Retrieved Sources"
                        ):

                            for i, doc in enumerate(
                                docs,
                                start=1
                            ):

                                page = doc.metadata.get(
                                    "page"
                                )

                                if page is not None:

                                    page_number = page + 1

                                    st.markdown(
                                        f"**Source {i} — "
                                        f"Page {page_number}**"
                                    )

                                else:

                                    st.markdown(
                                        f"**Source {i}**"
                                    )


                except Exception as e:

                    answer = (
                        "⚠️ I couldn't generate an answer "
                        "right now. Please try again."
                    )

                    st.error(
                        f"Error: {str(e)}"
                    )

                    st.markdown(answer)


        # ----------------------------------------------------
        # Save clean answer to chat history
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )