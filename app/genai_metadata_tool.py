"""
GenAI-Assisted Metadata Tool (Responsible AI)
Assistive intelligence, NOT autonomous decision-making
"""

# =========================
# 1. IMPORTS (2024+ SAFE)
# =========================
from typing import List
from pydantic import BaseModel, Field
import json
import sqlite3
import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# =========================
# 2. CONFIGURATION
# =========================
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.2
TOP_K = 5
SIMILARITY_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.2

ALLOWED_TAGS = {
    "revenge", "friendship", "betrayal", "survival",
    "love", "identity", "justice", "power"
}

DB_PATH = "audit_logs.db"
VECTORSTORE_PATH = "metadata_index"

# =========================
# 3. OUTPUT SCHEMA (Pydantic v2)
# =========================
class MetadataOutput(BaseModel):
    themes: List[str] = Field(..., max_length=3)
    mood: str
    confidence: float = Field(..., ge=0.0, le=1.0)

# =========================
# 4. AUDIT LOGGER
# =========================
def init_logger():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            context TEXT,
            output TEXT,
            confidence REAL,
            decision TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_event(query, context, output, confidence, decision):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO audit_logs
        VALUES (NULL, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.datetime.utcnow().isoformat(),
        query,
        context,
        json.dumps(output),
        confidence,
        decision
    ))
    conn.commit()
    conn.close()

# =========================
# 5. LOAD VECTOR STORE (SAFE)
# =========================
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True  # trusted local index
    )

# =========================
# 6. RETRIEVE + FILTER CONTEXT
# =========================
def retrieve_context(query, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.get_relevant_documents(query)

    filtered = [
        d.page_content
        for d in docs
        if d.metadata.get("similarity", 1.0) >= SIMILARITY_THRESHOLD
    ]

    return "\n".join(filtered)

# =========================
# 7. PROMPT (STRICT)
# =========================
PROMPT = PromptTemplate(
    template="""
You are assisting a content metadata editor.

Rules:
- Use ONLY the provided context
- Use ONLY the following allowed themes:
  revenge, friendship, betrayal, survival, love, identity, justice, power
- If information is missing, return exactly: "INSUFFICIENT CONTEXT"
- Suggest at most 3 themes
- Output MUST be valid JSON
- Do NOT include explanations or extra text

JSON schema:
{{
  "themes": ["string", "..."],
  "mood": "string",
  "confidence": 0.0
}}

Context:
{context}

Task:
Suggest themes and mood.
""",
    input_variables=["context"]
)

# =========================
# 8. LLM + PARSER
# =========================
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    model_kwargs={"response_format": {"type": "json_object"}}
)

parser = PydanticOutputParser(pydantic_object=MetadataOutput)

# =========================
# 9. VALIDATION
# =========================
def validate_output(output: MetadataOutput):
    if output.confidence < CONFIDENCE_THRESHOLD:
        return "REJECTED"

    for tag in output.themes:
        if tag not in ALLOWED_TAGS:
            return "REJECTED"

    return "APPROVED"

# =========================
# 10. MAIN PIPELINE
# =========================
def run_metadata_tool(query: str):
    vectorstore = load_vectorstore()
    context = retrieve_context(query, vectorstore)

    if not context.strip():
        log_event(query, "", {}, 0.0, "REJECTED")
        return "INSUFFICIENT CONTEXT"

    prompt_text = PROMPT.format(context=context)
    raw_output = llm.invoke(prompt_text)

    try:
        parsed = parser.parse(raw_output.content)
    except Exception:
        log_event(query, context, {}, 0.0, "REJECTED")
        return "REJECTED (PARSING ERROR)"

    decision = validate_output(parsed)

    log_event(
        query=query,
        context=context,
        output=parsed.model_dump(),
        confidence=parsed.confidence,
        decision=decision
    )

    return parsed if decision == "APPROVED" else "REJECTED"

# =========================
# 11. ENTRY POINT
# =========================
if __name__ == "__main__":
    init_logger()
    user_query = "Suggest themes and mood for a prison escape drama"
    result = run_metadata_tool(user_query)
    print(result)
