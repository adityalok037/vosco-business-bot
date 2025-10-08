"""
Embedding Service for Vector Search
Converts text to vector embeddings using Sentence Transformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from typing import List
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root (for imports if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class EmbeddingService:
    """
    Handles text to vector embedding conversion
    Uses all-MiniLM-L6-v2 model (384 dimensions, fast & accurate)
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize embedding model
        """
        try:
            logger.info(f"🔄 Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"✅ Model loaded successfully! Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """
        Convert single text to embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided, returning zero vector")
            return [0.0] * self.dimension

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            raise

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts to embeddings (faster than one-by-one)
        """
        try:
            embeddings = self.model.encode(
                texts, convert_to_numpy=True, show_progress_bar=True
            )
            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"❌ Batch embedding failed: {e}")
            raise

    def generate_embeddings_for_table(self, db, table: str, column: str, embedding_column: str):
        """
        Generate and store embeddings for existing table data

        Args:
            db: Database instance
            table: Table name
            column: Source column to embed
            embedding_column: Target embedding column
        """
        try:
            logger.info(f"🔄 Generating embeddings for {table}.{column} → {embedding_column}")

            # Check if embedding column exists
            check_query = f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = '{table}' AND column_name = '{embedding_column}'
            """
            if not db.execute_query(check_query):
                raise ValueError(
                    f"Embedding column '{embedding_column}' does not exist in table '{table}'. "
                    f"Add it via: ALTER TABLE {table} ADD COLUMN {embedding_column} vector({self.dimension});"
                )

            # Fetch only rows without embeddings
            query = f"""
            SELECT id, {column} 
            FROM {table} 
            WHERE {column} IS NOT NULL 
            AND {embedding_column} IS NULL
            """
            rows = db.execute_query(query)

            if not rows:
                logger.info(f"✅ No new embeddings needed for {table}.{embedding_column}")
                return

            logger.info(f"📊 Processing {len(rows)} new/missing records...")

            # Generate embeddings in batch
            texts = [row[column] for row in rows]
            embeddings = self.get_embeddings_batch(texts)

            # Prepare update data
            update_query = f"""
            UPDATE {table} 
            SET {embedding_column} = %s::vector 
            WHERE id = %s
            """
            update_data = [(emb, row['id']) for emb, row in zip(embeddings, rows)]

            # Batch update
            if hasattr(db, 'execute_many'):
                affected = db.execute_many(update_query, update_data)
                logger.info(f"✅ Batch updated {affected} embeddings in {table}")
            else:
                for params in update_data:
                    db.execute_query(update_query, params)
                logger.info(f"✅ Updated {len(update_data)} embeddings in {table} (single mode)")

        except Exception as e:
            logger.error(f"❌ Failed to generate embeddings for {table}: {e}")
            raise

    def similarity_score(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Cosine similarity between two embeddings
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


def initialize_all_embeddings(db):
    """
    Generate embeddings for all relevant tables
    """
    embedding_service = EmbeddingService()

    # Mapping: table, source_column, embedding_column
    tables_to_embed = [
        ('products', 'name', 'name_embedding'),
        ('orders', 'customer_name', 'customer_embedding')
    ]

    logger.info("🚀 Starting embedding generation for all tables...")

    for table, column, emb_col in tables_to_embed:
        try:
            embedding_service.generate_embeddings_for_table(db, table, column, emb_col)
        except Exception as e:
            logger.error(f"❌ Failed for {table}.{column} → {emb_col}: {e}")

    logger.info("✅ All embeddings generated successfully!")


# Test / CLI
if __name__ == "__main__":
    from database.connection import Database

    print("🔧 Embedding Service Test\n")

    service = EmbeddingService()

    # Single embedding
    test_text = "Dell Laptop"
    emb = service.get_embedding(test_text)
    print(f"✅ Single embedding: {test_text} → dim {len(emb)} first 5: {emb[:5]}")

    # Batch embedding
    test_texts = ["Laptop", "Mouse", "Keyboard"]
    embeddings = service.get_embeddings_batch(test_texts)
    print(f"✅ Batch embedding: {len(embeddings)} vectors for {test_texts}")

    # Similarity test
    emb1 = service.get_embedding("laptop computer")
    emb2 = service.get_embedding("notebook pc")
    emb3 = service.get_embedding("coffee mug")
    print(f"✅ Similarity similar: {service.similarity_score(emb1, emb2):.3f}")
    print(f"✅ Similarity different: {service.similarity_score(emb1, emb3):.3f}")

    # Generate database embeddings
    response = input("🔄 Generate embeddings for database tables? (y/n): ").strip().lower()
    if response == 'y':
        db = Database()
        initialize_all_embeddings(db)
        print("✅ Database embeddings generation complete!")
    else:
        print("⏭️ Skipped database embedding generation")
