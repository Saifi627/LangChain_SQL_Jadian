from dotenv import load_dotenv
import os
import urllib

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_SERVER = os.getenv("DB_SERVER")  # e.g. EPAZZ2\saifu
DB_NAME = os.getenv("DB_NAME")      # e.g. AAcounty

# Build the ODBC connection string safely
params = urllib.parse.quote_plus(
    f"Driver={{ODBC Driver 18 for SQL Server}};"
    f"Server={DB_SERVER};"
    f"Database={DB_NAME};"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

DB_URI = f"mssql+pyodbc:///?odbc_connect={params}"
