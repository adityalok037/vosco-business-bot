"""
Database Connection Manager with Connection Pooling
Production-ready PostgreSQL connection handler
"""

import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class Database:
    """
    Singleton database connection pool manager
    Uses connection pooling for better performance
    """
    _pool = None
    
    @classmethod
    def initialize_pool(cls):
        """Initialize connection pool once"""
        if cls._pool is None:
            try:
                cls._pool = SimpleConnectionPool(
                    minconn=1,
                    maxconn=10,
                    host=os.getenv('DB_HOST', 'localhost'),
                    port=os.getenv('DB_PORT', '5432'),
                    database=os.getenv('DB_NAME'),
                    user=os.getenv('DB_USER'),
                    password=os.getenv('DB_PASSWORD')
                )
                logger.info("✅ Database connection pool initialized")
            except Exception as e:
                logger.error(f"❌ Failed to create connection pool: {e}")
                raise
    
    @classmethod
    def get_pool(cls):
        """Get connection pool instance"""
        if cls._pool is None:
            cls.initialize_pool()
        return cls._pool
    
    @classmethod
    def get_connection(cls):
        """Get a connection from pool"""
        pool = cls.get_pool()
        return pool.getconn()
    
    @classmethod
    def return_connection(cls, conn):
        """Return connection to pool"""
        pool = cls.get_pool()
        pool.putconn(conn)
    
    @classmethod
    def execute_query(cls, query, params=None, fetch=True):
        """
        Execute SQL query with automatic connection management
        
        Args:
            query (str): SQL query to execute
            params (tuple): Query parameters for safe execution
            fetch (bool): Whether to fetch results (SELECT queries)
            
        Returns:
            list: Query results as list of dictionaries (for SELECT)
            None: For INSERT/UPDATE/DELETE operations
        """
        conn = None
        try:
            conn = cls.get_connection()
            
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Execute query
                cur.execute(query, params)
                
                # Check if it's a SELECT query
                if fetch and query.strip().upper().startswith('SELECT'):
                    results = cur.fetchall()
                    logger.info(f"✅ Query executed: {len(results)} rows returned")
                    return results
                else:
                    # For INSERT/UPDATE/DELETE
                    conn.commit()
                    logger.info("✅ Query executed successfully")
                    return None
                    
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"❌ Database error: {e}")
            raise
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"❌ Unexpected error: {e}")
            raise
        finally:
            if conn:
                cls.return_connection(conn)
    
    @classmethod
    def execute_many(cls, query, data_list):
        """
        Execute same query with multiple parameter sets
        Useful for batch inserts
        
        Args:
            query (str): SQL query with placeholders
            data_list (list): List of tuples containing parameters
        """
        conn = None
        try:
            conn = cls.get_connection()
            
            with conn.cursor() as cur:
                cur.executemany(query, data_list)
                conn.commit()
                logger.info(f"✅ Batch executed: {len(data_list)} operations")
                
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"❌ Batch execution error: {e}")
            raise
        finally:
            if conn:
                cls.return_connection(conn)
    
    @classmethod
    def test_connection(cls):
        """Test database connection"""
        try:
            result = cls.execute_query("SELECT version();")
            logger.info(f"✅ Database connected: {result[0]['version'][:50]}...")
            return True
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False
    
    @classmethod
    def get_table_stats(cls):
        """Get statistics of all tables"""
        query = """
        SELECT 
            'departments' as table_name, COUNT(*) as count FROM departments
        UNION ALL
        SELECT 'employees', COUNT(*) FROM employees
        UNION ALL
        SELECT 'products', COUNT(*) FROM products
        UNION ALL
        SELECT 'orders', COUNT(*) FROM orders
        ORDER BY table_name;
        """
        return cls.execute_query(query)
    
    @classmethod
    def close_all_connections(cls):
        """Close all connections in pool - call on app shutdown"""
        if cls._pool:
            cls._pool.closeall()
            logger.info("✅ All database connections closed")


# Quick test function
if __name__ == "__main__":
    print("Testing database connection...")
    
    # Test connection
    if Database.test_connection():
        print("✅ Connection successful!")
        
        # Get table stats
        stats = Database.get_table_stats()
        print("\n📊 Database Statistics:")
        for row in stats:
            print(f"  {row['table_name']}: {row['count']} records")
    else:
        print("❌ Connection failed! Check your .env file")