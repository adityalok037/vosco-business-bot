import streamlit as st
import pandas as pd
from collections import deque
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Services
from database.connection import Database
from services.embedding_service import EmbeddingService
from services.query_service import QueryService

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="🔍 VOSCO Bot NL Database Search",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- Session State -------------------
if 'chats' not in st.session_state:
    st.session_state.chats = {}  # chat_id -> deque of messages
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.chats[st.session_state.current_chat_id] = deque(maxlen=20)

if 'db' not in st.session_state:
    st.session_state.db = None
if 'embedding_service' not in st.session_state:
    st.session_state.embedding_service = None
if 'query_service' not in st.session_state:
    st.session_state.query_service = None
if 'search_mode' not in st.session_state:
    st.session_state.search_mode = "Hybrid"

# ------------------- Initialize Services -------------------
@st.cache_resource
def init_services():
    db = Database()
    embedding = EmbeddingService()
    query = QueryService()
    return db, embedding, query

if st.session_state.db is None:
    st.session_state.db, st.session_state.embedding_service, st.session_state.query_service = init_services()

db = st.session_state.db
embedding_service = st.session_state.embedding_service
query_service = st.session_state.query_service

# ------------------- Sidebar -------------------
with st.sidebar:
    st.header("💬 Chats")
    if st.button("🔄 New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = deque(maxlen=20)
        st.session_state.current_chat_id = new_id
        st.experimental_rerun()

    chat_ids = list(st.session_state.chats.keys())
    selected_chat = st.selectbox("Select Chat", chat_ids, index=chat_ids.index(st.session_state.current_chat_id))
    if selected_chat != st.session_state.current_chat_id:
        st.session_state.current_chat_id = selected_chat
        st.experimental_rerun()

    if st.button("🗑️ Delete This Chat"):
        del st.session_state.chats[st.session_state.current_chat_id]
        st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
        st.experimental_rerun()

    if st.button("🗑️ Delete All Chats"):
        st.session_state.chats = {str(uuid.uuid4()): deque(maxlen=20)}
        st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
        st.experimental_rerun()

    st.markdown("---")
    st.header("⚙️ Controls")
    st.session_state.search_mode = st.radio("Search Mode", ["SQL Query", "Vector Search", "Hybrid"])
    if st.session_state.search_mode in ["Vector Search", "Hybrid"]:
        st.session_state.vector_table = st.selectbox("Vector Table", ["products", "orders"])

# ------------------- Main UI -------------------
st.title("🔍 VOSCO Bot Natural Language Database Search")
st.markdown("Ask questions in plain English. Chat context is preserved per conversation.")

# Chat container
chat_container = st.container()
current_history = st.session_state.chats[st.session_state.current_chat_id]

# Display chat history
for msg in current_history:
    role = msg["role"]
    content = msg["content"]
    results = msg.get("results", None)
    with st.chat_message(role):
        st.markdown(content)
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

# Input
user_input = st.chat_input("Enter your question...")

if user_input:
    current_history.append({"role": "user", "content": user_input})
    with st.spinner("Processing..."):
        try:
            # Build context
            context = "\n".join([f"{m['role']}: {m['content']}" for m in list(current_history)[-5:]])
            enhanced_query = f"Context:\n{context}\nQuestion: {user_input}"

            results = None
            assistant_content = ""

            # ------------------- Hybrid Search -------------------
            if st.session_state.search_mode == "SQL Query":
                sql = query_service.nl_to_sql(enhanced_query)
                results = db.execute_query(sql)
                assistant_content = f"Found {len(results)} results via SQL." if results else "No results found."
            elif st.session_state.search_mode == "Vector Search":
                table = st.session_state.vector_table
                results = query_service.vector_search(db, enhanced_query, table, limit=5)
                assistant_content = f"Found {len(results)} similar results via vector search." if results else "No similar results found."
            else:  # Hybrid
                # Try SQL first
                try:
                    sql = query_service.nl_to_sql(enhanced_query)
                    results = db.execute_query(sql)
                    if results:
                        assistant_content = f"Found {len(results)} results via SQL."
                    else:
                        # Fallback vector search
                        table = st.session_state.vector_table
                        results = query_service.vector_search(db, enhanced_query, table, limit=5)
                        assistant_content = f"Found {len(results)} results via vector search (SQL empty)."
                except Exception:
                    table = st.session_state.vector_table
                    results = query_service.vector_search(db, enhanced_query, table, limit=5)
                    assistant_content = f"Found {len(results)} results via vector search (SQL failed)."

            # Append assistant message
            msg = {"role": "assistant", "content": assistant_content}
            if results:
                msg["results"] = results
            current_history.append(msg)

        except Exception as e:
            current_history.append({"role": "assistant", "content": f"Error: {str(e)}"})

    st.rerun()
