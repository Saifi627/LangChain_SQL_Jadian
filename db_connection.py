from sqlalchemy import create_engine
from config import DB_URI

def get_engine():
    engine = create_engine(DB_URI)
    return engine


