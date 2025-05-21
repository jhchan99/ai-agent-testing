from dotenv import load_dotenv
from typing_extensions import TypedDict, Any
from agents import Agent, FunctionTool, RunContextWrapper, function_tool, Runner
from pydantic import BaseModel
from pinecone import Pinecone
import asyncio
from openai import OpenAI
import os

# Load environment variables from .env file
load_dotenv()
# Initialize Pinecone
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


# this agent will triage the question to the appropriate agent
officer_agent = Agent(
    name="Officer Agent",
    instructions="You are an assistant that uses tools to answer queries. You will use the correct tools to answer the queries.",
    tools=[fetch_sworn_content, fetch_agency_content, fetch_officer_content],
    # input_guardrails=[
    #     InputGuardrail(guardrail_function=homework_guardrail),
    # ],
)


async def main():
    """
    This is the main function that runs the triage agent with different inputs.
    """
    result = await Runner.run(
        officer_agent,
        "what are my benefits for a death in my family?",
    )
    print(f"Answer: {result.final_output}")
    print(f"Thoughts: {result.thoughts}")


if __name__ == "__main__":
    asyncio.run(main())
