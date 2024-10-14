import streamlit as st
from streamlit import session_state
import time
import base64
import os
from embeddings import EmbeddingsManager  # Import the EmbeddingsManager class
from bot import ChatbotManager     # Import the ChatbotManager class

# --- Configuration Section ---
# Centralized configuration to hold all static values
CONFIG = {
    "embedding_model": "BAAI/bge-small-en",
    "device": "cpu",
    "encode_kwargs": {"normalize_embeddings": True},
    "qdrant_url": "http://localhost:6333",
    "collection_name": "vector_db",
    "llm_model": "llama3.2:3b",
    "llm_temperature": 0.7,
}

# --- Helper Functions ---
def display_pdf(file):
    """
    Display the given PDF file in the Streamlit UI.
    
    Args:
        file: Uploaded PDF file.
    """
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def initialize_embeddings_manager():
    """
    Initialize the EmbeddingsManager with predefined configurations.
    
    Returns:
        An instance of EmbeddingsManager.
    """
    return EmbeddingsManager(
        model_name=CONFIG["embedding_model"],
        device=CONFIG["device"],
        encode_kwargs=CONFIG["encode_kwargs"],
        qdrant_url=CONFIG["qdrant_url"],
        collection_name=CONFIG["collection_name"]
    )


def initialize_chatbot_manager():
    """
    Initialize the ChatbotManager with predefined configurations.
    
    Returns:
        An instance of ChatbotManager.
    """
    return ChatbotManager(
        model_name=CONFIG["embedding_model"],
        device=CONFIG["device"],
        encode_kwargs=CONFIG["encode_kwargs"],
        llm_model=CONFIG["llm_model"],
        llm_temperature=CONFIG["llm_temperature"],
        qdrant_url=CONFIG["qdrant_url"],
        collection_name=CONFIG["collection_name"]
    )


# --- Session State Initialization ---
if 'state' not in session_state:
    session_state['state'] = {
        'temp_pdf_path': None,
        'chatbot_manager': None,
        'messages': []
    }

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Information Extractor App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar Navigation ---
with st.sidebar:
    st.image("bot.png", use_column_width=True)
    st.markdown("### 📚 Your Personal Document Assistant")
    st.markdown("---")
    menu = ["🏠 Home", "🤖 Chatbot", "📧 Contact"]
    choice = st.selectbox("Navigate", menu)

# --- Home Page ---
if choice == "🏠 Home":
    st.title("📄 Document ChatBot")
    st.markdown("""
    Welcome to **Document ChatBot**! 🚀

    **Built using Open Source Stack (Llama 3.2, BGE Embeddings, and Qdrant running locally within a Docker Container.)**

    - **Upload Documents**: Easily upload your PDF documents.
    - **Summarize**: Get concise summaries of your documents.
    - **Chat**: Interact with your documents through our intelligent chatbot.

    Enhance your document management experience with Document Bot! 😊
    """)

# --- Chatbot Page ---
elif choice == "🤖 Chatbot":
    st.title("🤖 Your Favourite Information Extractor (Llama 3.2 RAG 🦙)")
    st.markdown("---")
    
    # Create three columns
    col1, col2, col3 = st.columns(3)

    # Column 1: File Uploader and Preview
    with col1:
        st.header("📂 Upload Document")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file:
            st.success("📄 File Uploaded Successfully!")
            st.markdown(f"**Filename:** {uploaded_file.name}")
            st.markdown(f"**File Size:** {uploaded_file.size} bytes")

            # Display PDF preview using displayPDF function
            st.markdown("### 📖 PDF Preview")
            display_pdf(uploaded_file)

            # Save the uploaded file to a temporary location
            temp_pdf_path = "doc.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            session_state['state']['temp_pdf_path'] = temp_pdf_path

    # Column 2: Create Embeddings
    with col2:
        st.header("🧠 Embeddings")
        create_embeddings = st.checkbox("✅ Create Embeddings")
        if create_embeddings:
            if not session_state['state']['temp_pdf_path']:
                st.warning("⚠️ Please upload a PDF first.")
            else:
                try:
                    embeddings_manager = initialize_embeddings_manager()
                    with st.spinner("🔄 Embeddings are in process..."):
                        result = embeddings_manager.create_embeddings(session_state['state']['temp_pdf_path'])
                        time.sleep(1)
                    st.success(result)

                    if not session_state['state']['chatbot_manager']:
                        session_state['state']['chatbot_manager'] = initialize_chatbot_manager()

                except FileNotFoundError as fnf_error:
                    st.error(fnf_error)
                except ValueError as val_error:
                    st.error(val_error)
                except ConnectionError as conn_error:
                    st.error(conn_error)
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    # Column 3: Chatbot Interface
    with col3:
        st.header("💬 Chat with Document")
        
        if not session_state['state']['chatbot_manager']:
            st.info("🤖 Please upload a PDF and create embeddings to start chatting.")
        else:
            for msg in session_state['state']['messages']:
                st.chat_message(msg['role']).markdown(msg['content'])

            if user_input := st.chat_input("Type your message here..."):
                st.chat_message("user").markdown(user_input)
                session_state['state']['messages'].append({"role": "user", "content": user_input})

                with st.spinner("🤖 Responding..."):
                    try:
                        answer = session_state['state']['chatbot_manager'].get_response(user_input)
                        time.sleep(1)
                    except Exception as e:
                        answer = f"⚠️ An error occurred while processing your request: {e}"
                
                st.chat_message("assistant").markdown(answer)
                session_state['state']['messages'].append({"role": "assistant", "content": answer})

# --- Contact Page ---
elif choice == "📧 Contact":
    st.title("📬 Contact Us")
    st.markdown("""
    Would love to hear from you! Whether you have a question, feedback, or want to contribute, feel free to reach out.

    - **Email:** [developer@example.com](mailto:damilolagboye@gmail.com) ✉️
    
    """)

# --- Footer ---
st.markdown("---")
st.markdown("...Document ChatBot by Cartel. 🛡️")