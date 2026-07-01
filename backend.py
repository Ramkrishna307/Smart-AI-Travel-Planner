import os
import requests
from dotenv import load_dotenv

os.environ["SSL_CERT_FILE"]=certifi.where()
os.environ["REQUESTS_CA_BUNDL"]=certifi.where()

from typing import TypeDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START,END
from langgraph.checkpoint.postgres import PostgressSaaver
from langchain_core.messsages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)


from langchain.groq import ChatGroq
from tools.tavily import tavily_search
from tools.flight import flight_search


load_dotenv()



def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")



# =========================
# LLM
# =========================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)


