from dotenv import load_dotenv
from typing import Optional, List, Dict
from typing_extensions import TypedDict, Any
from agents import Agent, FunctionTool, RunContextWrapper, function_tool, Runner
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from pydantic import BaseModel
from pinecone import Pinecone
import asyncio
from datetime import datetime
from openai import OpenAI
import os
from tools.health_tools import (
    fetch_sworn_content,
    fetch_agency_content,
    fetch_officer_content,
    fetch_officer_fitness_data,
    fetch_officer_CAD_data,
    fetch_officer_nutrition_data,
    fetch_sleep_data,
)

# Load environment variables from .env file
load_dotenv()
# Initialize Pinecone
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("sworn-pinecone-index")
now = datetime.now().strftime("%Y-%m-%d")

# this agent will have access to officer health data apis
health_agent = Agent(
    name="Health Agent",
    model="gpt-4o-mini",
    instructions=RECOMMENDED_PROMPT_PREFIX
    + "The current date and time is"
    + f"{now}"
    + "You are a health agent. You will make calls to the health api to get the officers health data given a date range. Consider the current date and time when making the call."
    + "You will be given a general query and you will need to determine wchich tools to use, you may need to make multiple calls to different tools to answer the query.",
    tools=[
        fetch_officer_fitness_data,
        fetch_officer_CAD_data,
        fetch_officer_nutrition_data,
        fetch_sleep_data,
    ],
)

# this agent will use tools and other agents to answer queries
content_agent = Agent(
    name="Content Agent",
    instructions=RECOMMENDED_PROMPT_PREFIX
    + "You are an assistant that uses tools to retrieve content from the database. You will use the correct tools to answer the queries.",
    tools=[fetch_sworn_content, fetch_agency_content, fetch_officer_content],
    # input_guardrails=[
    #     InputGuardrail(guardrail_function=homework_guardrail),
    # ],
)

# triage agent will decide which agent to handoff to
triage_agent = Agent(
    name="Triage Agent",
    instructions="You are a triage agent. You will decide which agent to handoff to based on the query.",
    handoffs=[content_agent, health_agent],
)


async def main():
    """
    This is the main function that runs the triage agent with different inputs.
    """
    result = await Runner.run(
        triage_agent,
        "how has my work calls been affecting my sleep?",
    )
    print(f"Answer: {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
