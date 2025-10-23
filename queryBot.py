# queryBot.py
# -------------------------------------------------------
# 💪 Reliable SQL Agent + Summarization handoff (Gemini)
# -------------------------------------------------------

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.tools import tool
from db_connection import get_engine
from config import GEMINI_API_KEY


# 1️⃣ Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
    verbose=True,
)

# 2️⃣ Connect to SQL DB
engine = get_engine()
db = SQLDatabase(engine, include_tables=["AACountyexcel"])

# 3️⃣ Schema Context (minimal version)
schema_context = """
You are an expert assistant helping health inspectors query a Microsoft SQL Server database.
The database contains inspection, permit, and facility information for food establishments in Anne Arundel County.
Only use the table [AACountyexcel].
All column names contain spaces and must be wrapped in [square brackets].
Use only SELECT statements — never INSERT, UPDATE, or DELETE.
You write SQL queries ONLY for **Microsoft SQL Server (T-SQL)**.
Never use MySQL or PostgreSQL syntax.

Formatting rules:
- NEVER include backticks (`) or Markdown code fences (```sql ... ```).
- NEVER prefix or suffix queries with ``` or quotes.
- Write the query as raw SQL text, not Markdown.
- Always wrap column and table names in [square brackets], like [FSF NAME] or [AACountyexcel] when making sql queries. 
- End each query with a semicolon only if needed.
- Output ONLY the SQL code — no explanations, no commentary.

Column hints for AAcountyexcel table:
[FSF#] -> Facility’s unique identifier number.
[FSF NAME] -> Official facility name (e.g., restaurant or grocery store name).
[FSF ADDRESS] -> Street address of the facility.
[FSF CITY] -> City in which the facility is located.
[FSF STATE] -> U.S. state of the facility.
[FSF ZIP] -> Facility ZIP/postal code.
[FSF PHONE#] -> Facility contact phone number.
[BUSINESS NAME] -> Registered business name or legal entity of the facility.
[OWNERS FIRST NAME] -> First name of the facility owner.
[OWNERS LAST NAME] -> Last name of the facility owner.
[OWNERS ADDRESS] -> Owner’s home or mailing address.
[OWNERS CITY] -> City for the owner’s address.
[OWNER STATE] -> State for the owner’s address.
[OWNERS ZIP] -> ZIP/postal code for the owner’s address.
[OWNERS PHONE] -> Owner’s primary phone number.
[OWNERS E-MAIL] -> Owner’s email address.
[CONTACT NAME] -> Primary contact person for the facility (not always the owner).
[CONTACT PHONE] -> Contact person’s phone number.
[CONTACT E-MAIL] -> Contact person’s email address.
[AREA] -> Geographic inspection or service area.
[FACILITY TYPE] -> Category of facility (e.g., restaurant, school kitchen, retail food store).
[RISK] -> Risk classification level (1 = high, 2 = medium, 3 = low).
[SEASONAL] -> Indicates if the facility operates seasonally (e.g., open only in summer).
[EXEMPT] -> Whether the facility is exempt from certain permit requirements.
[CITY OF ANNAPOLIS] -> Flag indicating if the facility is within Annapolis city limits.
[PERMIT ISSUE DATE] -> Date when the operating permit was issued.
[PERMIT EXPIRATION DATE] -> Date when the permit expires.
[FACILITY SEATING NUMBER] -> Number of seats available for customers.
[SEPTIC DESIGN SEATING#] -> Number of seats approved in septic design.
[WATER] -> Water supply type (e.g., public, private well).
[SEWER] -> Sewer type (e.g., public, septic).
[GREAT TRAP] -> Likely “grease trap” — indicates if grease control equipment is installed.
[GREASE RECOVERY] -> Indicates if grease recovery system is used.
[COMMENT 1] -> Inspector comments or notes.
[TAX ID] -> Tax identification number for the business.
[INSPECTOR#] -> Unique ID for assigned inspector.
[PLAN REVIEW FEE] -> Fee for reviewing facility construction/renovation plans.
[PLAN REVIEW PAID DATE] -> Date plan review fee was paid.
[HACCP FEE] -> Fee associated with HACCP (Hazard Analysis Critical Control Point) review.
[HACCP PAID DATE] -> Date HACCP fee was paid.
[APPLICATION FEE] -> Fee for the facility’s permit application.
[APPLICATION PAID DATE] -> Date the application fee was paid.
[APPLICATION REC DATE] -> Date the application was received.
[OPEN DATE] -> Date facility was opened for business.
[CLOSED DATE] -> Date facility was permanently closed (if applicable).
[CI INSP DATE] -> Date of the initial Critical Item inspection.
[CI INSPECTOR #] -> Inspector ID for the Critical Item inspection.
[Critical Item CODE] -> Code describing the specific critical violation observed.
[CI RE DATE] -> Date of first re-inspection after a critical violation.
[CI RE INSPECTOR #] -> Inspector ID for the re-inspection.
[Critical Item RE CODE] -> Code describing the issue during re-inspection.
[CI RE2 DATE] -> Date of second re-inspection for a critical violation.
[CI RE2 ISPECTOR #] -> Inspector ID for second re-inspection.
[Critical ItemRE2 CODE] -> Violation code for second re-inspection.
[CIRE3 DATE] -> Date of third re-inspection (if any).
[CI RE3 INSPECTOR #] -> Inspector ID for third re-inspection.
[Critical Item R3 CODE] -> Violation code during third re-inspection.
[CIRE NEEDED] -> Indicates if additional re-inspection is needed.
[M1 INSPECTION DATE] -> Date of the first “M1” (routine) inspection.
[M1 INSPECTOR#] -> Inspector ID for M1 inspection.
[M1 INSP Critical Item CODE] -> Violation codes found in M1 inspection.
[M1 RE DATE] -> Date of first M1 re-inspection.
[M1 RE INSPECTOR#] -> Inspector ID for first M1 re-inspection.
[M1 RE Critical Item CODE] -> Violation codes found during first M1 re-inspection.
[M1 RE2 DATE] -> Date of second M1 re-inspection.
[M1 RE2 INSPECTOR#] -> Inspector ID for second M1 re-inspection.
[M1RE 2 Critical Item CODE] -> Violation codes during second M1 re-inspection.
[M1 RE3 DATE] -> Date of third M1 re-inspection.
[M1 RE3 INSPECTOR#] -> Inspector ID for third M1 re-inspection.
[M1RE3 Critical Item CODE] -> Violation codes during third M1 re-inspection.
[M1RE NEEDED] -> Indicates if further M1 re-inspection is needed.
[M2 INSP DATE] -> Date of M2 inspection.
[M2 INSP#] -> Inspector ID for M2 inspection.
[M2 Critical Item CODE] -> Violation codes for M2 inspection.
[M2 RE DATE] -> Date of M2 re-inspection.
[M2 RE INSP#] -> Inspector ID for M2 re-inspection.
[M2 RE Critical Item CODE] -> Violation codes during M2 re-inspection.
[M2 RE2 DATE] -> Date of second M2 re-inspection.
[M2 RE2 ISNPEC#] -> Inspector ID for second M2 re-inspection.
[M2 R2 Critical Item CODE] -> Violation codes during second M2 re-inspection.
[M2 RE3 DATE] -> Date of third M2 re-inspection.
[M2 RE3 INSP#] -> Inspector ID for third M2 re-inspection.
[M2RE3 Critical Item CODE] -> Violation codes during third M2 re-inspection.
[M2 RE NEEDED] -> Indicates if further M2 re-inspection is needed.
[BO#] -> Possibly “back office” or “business office” ID (unclear).
[HACCP update] -> Notes or date of last HACCP plan update.
[HACCP update inspector#] -> Inspector responsible for HACCP update.
[HACCP update inspector name] -> Name of HACCP update inspector.
[F89] -> (unknown purpose)
[inserted] -> System field marking when the record was inserted.
[itemid] -> (unknown purpose)
[addressitem] -> Possibly linked address reference key.
[insertednum] -> Internal insertion sequence or counter.
"""

# 4️⃣ Define a summarization tool
@tool("summarize_data", return_direct=True)
def summarize_data(data: str) -> str:
    """Summarizes SQL query results into a human-readable response."""
    summary_prompt = f"""
    You are a helpful data summarizer.
    The following is output from a SQL query about health facilities:

    {data}

    Write a concise summary highlighting:
    - Facility name and address
    - Permit issue/expiry dates
    - Risk level or facility type if present
    - Avoid verbose explanations.
    """
    summary = llm.invoke(summary_prompt)
    return summary.content


# 5️⃣ Create SQL Agent
agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    system_message=schema_context,
)

# 6️⃣ Run and summarize manually
def query_and_summarize(user_query: str):
    """Runs the SQL agent, gets query output, then summarizes it."""
    print(f"\n🚀 Running QueryBot for: {user_query}\n")

    # Step 1: Ask agent to execute SQL
    query_result = agent.invoke({"input": user_query})
    raw_output = query_result.get("output", "")

    # Step 2: Summarize if output looks like SQL data
    if "SELECT" not in raw_output.upper() and len(raw_output) > 20:
        return f"🧠 Summary:\n{raw_output}"
    else:
        # Agent only generated query, not data — ask summarizer
        summary = summarize_data(raw_output)
        return f"🧠 Summary:\n{summary}"


# 7️⃣ Example usage
if __name__ == "__main__":
    question = "Show me info for Riva Food Market."
    answer = query_and_summarize(question)
    print("\n✅ Final Answer:\n", answer)
