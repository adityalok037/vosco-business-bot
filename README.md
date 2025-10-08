# VOSCO Bot: Advanced Natural Language Database Search Interface

A robust, production-grade Streamlit application enabling **natural language queries** on a PostgreSQL database. It integrates **AI-driven SQL generation**, **vector-based semantic search**, and a modern **chat interface** for intuitive data exploration.

---

## Executive Summary

This project delivers a complete prototype for querying structured data via plain English questions, aligning with requirements for **accuracy, usability, and efficiency**. Key components include:

- **Database Setup:** PostgreSQL schema with relationships and vector embeddings.
- **Search Capabilities:** NL-to-SQL conversion, vector search, and hybrid modes.
- **User Interface:** Chat-inspired design with contextual history and multi-chat support.
- **Deliverables:** Source code, setup guide, tests, and improvement strategies.
- **Highlights:** Modular code, secure query validation, and scalable features for real-world deployment.

For a quick demo, run the app locally or view the screen recording (link in Deliverables section).

---

## Project Overview

This application provides a **natural language search interface** for a PostgreSQL database managing **employee, department, order, and product information**.  
It uses:

- **Gemini AI** for converting queries to SQL  
- **pgvector** for semantic vector search  
- **Streamlit** for an interactive, chat-based UI  

The system supports:

- Contextual follow-up questions  
- Multiple conversation threads  
- Hybrid search (SQL + vector) for reliable results

---

## Core Features

### 1. AI-Powered Query Translation (NL → SQL)
- **How it Works:** User questions are sent to **Gemini 2.5 Flash**, generating a safe SELECT query with schema awareness.  
- **Highlights:** Handles joins, aggregations (COUNT, SUM, AVG), and semantic hints for vector use.  
- **Security:** Read-only queries only; prevents SQL injection.  
- **Code Location:** `services/query_service.py` (`nl_to_sql` method)  

### 2. Semantic Vector Search
- **How it Works:** Text fields are embedded using **Sentence Transformers**. Searches use **cosine similarity** to find relevant results.  
- **Highlights:** Supports fuzzy matching (e.g., "gaming laptop" → "Dell Latitude"), top N results, and table selection (`products`, `orders`).  
- **Code Location:** `services/embedding_service.py`, `query_service.py` (`vector_search` method)  

### 3. Hybrid Search Mode
- **How it Works:** Tries **NL-to-SQL first**; if empty or fails, falls back to **vector search**.  
- **Benefit:** Combines **exact matches** with **semantic matches** for robust query handling.  
- **Code Location:** `app.py` (user input handler)

---

## User Interface (UI) Features

### 1. Chat Interface
- **Design:** Right-aligned user input, left-aligned assistant responses with **data tables**.  
- **Scrollable history:** Supports seamless follow-ups.  
- **Code Location:** `app.py` (`chat_container`)

### 2. Contextual Query Handling
- Maintains per-chat history (**deque**, max 20 messages).  
- Includes last 5 messages in prompts for **context-aware responses**.  
- **Benefit:** Supports multi-turn interactions without repeating queries.  

### 3. Multi-Chat Management
- Sidebar features: **New Chat**, **Delete This Chat**, **Delete All Chats**, **Switch Chat**  
- **Benefit:** Users can manage multiple query threads simultaneously.  

### 4. Search Mode Selection
- Options: **SQL**, **Vector**, **Hybrid**  
- Dynamic **vector table selection** (`products`, `orders`) for semantic search.  
- **Benefit:** Provides flexible query strategies for users.

---

## Database and Backend Features

### 1. Connection Pooling
- Uses `psycopg2` **SimpleConnectionPool** for efficient DB access  
- Handles high concurrency and batch operations  
- **Code Location:** `database/connection.py`  

### 2. Embedding Generation
- 384-dimensional vectors for text fields  
- Updates only missing embeddings  
- **Code Location:** `services/embedding_service.py`  

### 3. Query Validation & Logging
- `is_safe_query` method blocks destructive SQL operations  
- Detailed logging for queries and errors  
- Ensures **production security and traceability**

---

## Testing and Development Features

- **Unit Tests:** Pytest coverage for NL-to-SQL and vector search, including error handling  
- **Caching & Session State:** `@st.cache_resource` for services, `session_state` for chats/modes  
- **Benefit:** Boosts performance in Streamlit's reactive environment  

---

## Additional Utilities

- **Sample Queries Expander:** Provides quick-start examples  
- **Error Handling:** Graceful display of failures in chat  
- **Rerun Mechanism:** Uses `st.experimental_rerun()` for instant UI updates  

---

## Setup Instructions

### Prerequisites
- Python 3.8+  
- PostgreSQL 14+ with pgvector extension  
- Gemini API key  
- Git, pip, virtualenv  

### Installation


git clone ([https://github.com/adityalok037/vosco-business-bot/])
cd vosco-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
