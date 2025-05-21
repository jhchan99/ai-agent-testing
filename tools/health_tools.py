from dotenv import load_dotenv
from typing import Optional, List, Dict
from typing_extensions import Any
from agents import function_tool
# from pydantic import BaseModel
from pinecone import Pinecone
import csv
# from datetime import datetime
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("sworn-pinecone-index")


async def generate_embedding(text):
    """Generate embedding for search query"""
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


@function_tool
async def fetch_sworn_content(query: str) -> Any:
    """
    this function retrieves sworn.ai company content. Content curated specifically to help give police officers helpful answers to their questions.
    """
    embedding = await generate_embedding(query)
    results = index.query(
        namespace="sworn-namespace",
        vector=embedding,
        top_k=3,
        include_metadata=True,
        include_values=False,
    )
    return results["matches"]


@function_tool
async def fetch_agency_content(query: str) -> Any:
    """
    this function retrieves content from the agency namespace. this is content that is specific to the agency and the needs of their officers. this could be agency policies, procedures, or other content that is specific to the agency.
    """
    embedding = await generate_embedding(query)
    results = index.query(
        namespace="FL0580300-northportnamespace",
        vector=embedding,
        top_k=3,
        include_metadata=True,
        include_values=False,
    )
    return results["matches"]


@function_tool
async def fetch_officer_content(query: str) -> Any:
    """
    this function is for retrieving content from the officers personal content namespace. this is content that is specific to the officer and their needs.
    """
    embedding = await generate_embedding(query)
    results = index.query(
        namespace="WI0680500-Waukesha-dthompson@waukesha-wi.gov",
        vector=embedding,
        top_k=3,
        include_metadata=True,
        include_values=False,
    )
    return results["matches"]


@function_tool
async def fetch_sleep_data(
    start_date: Optional[str] = None, end_date: Optional[str] = None
) -> List[Dict]:
    """
    Fetch sleep data for the officer within a date range.

    Args:
        start_date: The start date (YYYY-MM-DD) for the data range.
        end_date: The end date (YYYY-MM-DD) for the data range.
    """
    path = os.path.join("mock_data", "sleep.csv")
    results = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_date = row.get("date")
            if row_date:
                if (not start_date or row_date >= start_date) and (
                    not end_date or row_date <= end_date
                ):
                    results.append(row)
    return results


@function_tool
async def fetch_officer_fitness_data(
    start_date: Optional[str] = None, end_date: Optional[str] = None
) -> List[Dict]:
    """
    Fetch fitness data for the officer within a date range.

    Args:
        start_date: The start date (YYYY-MM-DD) for the data range.
        end_date: The end date (YYYY-MM-DD) for the data range.
    """
    path = os.path.join("mock_data", "fitness.csv")
    results = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_date = row.get("date")
            if row_date:
                if (not start_date or row_date >= start_date) and (
                    not end_date or row_date <= end_date
                ):
                    results.append(row)
    return results


@function_tool
async def fetch_officer_CAD_data(
    start_date: Optional[str] = None, end_date: Optional[str] = None
) -> List[Dict]:
    """
    Fetch CAD data for the officer within a date range.

    Args:
        start_date: The start date (YYYY-MM-DD) for the data range.
        end_date: The end date (YYYY-MM-DD) for the data range.
    """
    path = os.path.join("mock_data", "CAD.csv")
    results = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_date = row.get("date")
            if row_date:
                if (not start_date or row_date >= start_date) and (
                    not end_date or row_date <= end_date
                ):
                    results.append(row)
    return results


@function_tool
async def fetch_officer_nutrition_data(
    start_date: Optional[str] = None, end_date: Optional[str] = None
) -> List[Dict]:
    """
    Fetch nutrition data for the officer within a date range.

    Args:
        start_date: The start date (YYYY-MM-DD) for the data range.
        end_date: The end date (YYYY-MM-DD) for the data range.
    """
    path = os.path.join("mock_data", "nutrition.csv")
    results = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_date = row.get("date")
            if row_date:
                if (not start_date or row_date >= start_date) and (
                    not end_date or row_date <= end_date
                ):
                    results.append(row)
    return results
