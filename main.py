from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# Hugging Face Embedding Model
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# Chroma Vector Database
# ============================================================

vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory="chroma_db"
)


# ============================================================
# Retriever
# ============================================================

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)


# ============================================================
# Gemini LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)


# ============================================================
# Prompt
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say exactly:

"I could not find the answer in the document."

Keep your answer clear, concise, and easy to understand.
Do not return JSON, Python objects, metadata, signatures,
or any internal information."""
        ),
        (
            "human",
            """Context:

{context}

Question:

{question}
"""
        )
    ]
)


# ============================================================
# Clean Gemini/LangChain Response
# ============================================================

def extract_response_text(response):
    """
    Extract only the actual text from a LangChain AIMessage.

    Some Gemini/LangChain versions return response.content
    as a list of content blocks instead of a plain string.
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
            # {"type": "text", "text": "actual answer", ...}
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text", "")

                    if text:
                        text_parts.append(text)

            # In case a plain string appears inside the list
            elif isinstance(block, str):
                text_parts.append(block)

        return "\n".join(text_parts).strip()

    # Fallback
    return str(content).strip()


# ============================================================
# RAG System Ready
# ============================================================

print("RAG system created successfully.")
print("Type 0 to exit.")


# ============================================================
# Chat Loop
# ============================================================

while True:

    query = input("\nYou: ")

    if query.strip() == "0":
        print("Exiting...")
        break

    if not query.strip():
        print("Please enter a question.")
        continue

    # --------------------------------------------------------
    # Retrieve relevant documents
    # --------------------------------------------------------

    docs = retriever.invoke(query)

    if not docs:
        print("\nAI: I could not find relevant information in the document.")
        continue

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # --------------------------------------------------------
    # Create final prompt
    # --------------------------------------------------------

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    try:

        response = llm.invoke(final_prompt)

        # Extract only clean text
        answer = extract_response_text(response)

        print("\nAI:")
        print(answer)

        # ----------------------------------------------------
        # Show retrieved sources
        # ----------------------------------------------------

        print("\nSources:")

        for i, doc in enumerate(docs, start=1):

            page = doc.metadata.get("page")

            if page is not None:
                page_number = page + 1
                print(f"  {i}. Page {page_number}")
            else:
                print(f"  {i}. Document chunk {i}")

        print()

    except Exception as e:

        print("\nAI Error:")
        print(str(e))
        print()