import pytest
import sys
import os


# Project root (for imports if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
Production-ready Query Service
- Natural Language to SQL via Gemini
- Vector Search via PostgreSQL + pgvector
- Safe queries, logging, batch embedding support
"""

import os
import re
import logging
from typing import List, Dict
from dotenv import load_dotenv
import google.generativeai as genai
from services.embedding_service import EmbeddingService
from pgvector.psycopg2 import register_vector

# Simplified logging to avoid Windows handle issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))


class QueryService:
    """
    Converts natural language queries to SQL and handles vector search
    """

    table_config = {
        'products': {'column': 'name', 'embedding_column': 'name_embedding'},
        'orders': {'column': 'customer_name', 'embedding_column': 'customer_embedding'}
    }

    schema = """
    Tables:
    1. employees (id, name, department_id, email, salary)
    2. departments (id, name)
    3. orders (id, customer_name, employee_id, order_total, order_date, customer_embedding)
    4. products (id, name, price, name_embedding)

    Relationships:
    - employees.department_id -> departments.id
    - orders.employee_id -> employees.id

    Notes:
    - Use JOINs for related tables
    - For semantic search, use vector columns: products.name_embedding, orders.customer_embedding
    - Use <=> for cosine distance in vector search
    """

    def __init__(self):
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model
            self.embedding_service = EmbeddingService()
            logger.info("✅ QueryService initialized with Gemini model")
        except Exception as e:
            logger.error(f"❌ Failed to initialize QueryService: {e}")
            raise

    def nl_to_sql(self, question: str) -> str:
        """
        Convert natural language question to SQL using Gemini
        """
        try:
            prompt = f"""
            Database Schema:
            {self.schema}

            Convert this question to a PostgreSQL SELECT query:
            "{question}"

            Rules:
            - Return ONLY the SQL query
            - Use JOINs when needed
            - Proper WHERE clauses
            - No destructive operations (DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, TRUNCATE)
            - Use vector columns with <=> for semantic searches if the question implies similarity matching
            - Handle aggregations (COUNT, SUM, AVG)
            """
            response = self.model.generate_content(prompt)
            sql = response.text.strip()
            sql = re.sub(r'```sql')
            sql = re.sub(r'```\n?', '', sql)
            sql = sql.strip()

            if not self.is_safe_query(sql):
                raise ValueError("Unsafe query detected")

            logger.info(f"✅ Generated SQL: {sql}")
            return sql

        except Exception as e:
            logger.error(f"❌ Failed to generate SQL: {e}")
            raise

    def is_safe_query(self, query: str) -> bool:
        """
        Ensure query contains no destructive operations
        """
        dangerous = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        return not any(word in query.upper() for word in dangerous)

    def vector_search(self, db, search_text: str, table: str, limit: int = 5) -> List[Dict]:
        """
        Perform vector search on products or orders
        """
        try:
            if table not in self.table_config:
                raise ValueError(f"Invalid table '{table}'. Must be one of {list(self.table_config.keys())}")

            config = self.table_config[table]
            column = config['column']
            emb_col = config['embedding_column']

            embedding = self.embedding_service.get_embedding(search_text)

            pool = db.get_pool()
            conn = pool.getconn()
            try:
                register_vector(conn)
                query = f"""
                SELECT *, {emb_col} <=> %s::vector as distance
                FROM {table}
                ORDER BY distance
                LIMIT %s
                """
                results = db.execute_query(query, (embedding, limit))
            finally:
                pool.putconn(conn)

            logger.info(f"✅ Vector search on {table}: {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"❌ Vector search failed: {e}")
            raise


if __name__ == "__main__":
    from database.connection import Database

    db = Database()
    qs = QueryService()

    # NL-to-SQL test
    questions = [
        "Show all employees in Engineering department",
        "What is the average salary by department?",
        "List all orders above 50000"
    ]
    for q in questions:
        try:
            sql = qs.nl_to_sql(q)
            print(f"Question: {q}\nSQL: {sql}\n")
        except Exception as e:
            print(f"❌ Failed for '{q}': {e}\n")

    # Vector search test
    searches = [
        ('products', 'gaming laptop'),
        ('orders', 'Ravi')
    ]

    for table, text in searches:
        try:
            results = qs.vector_search(db, text, table, limit=3)
            print(f"Vector search: '{text}' in {table} ({len(results)} results)")
            for row in results:
                col = qs.table_config[table]['column']
                print(f"  - {row[col]} (distance: {row['distance']:.3f})")
            print()
        except Exception as e:
            print(f"❌ Failed vector search for '{text}' in {table}: {e}\n")