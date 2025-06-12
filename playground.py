from agno.agent import Agent, RunResponse
from agno.models.groq import Groq
from agno.playground import Playground, serve_playground_app
from agno.storage.sqlite import SqliteStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
import os
import groq

agent_storage: str = "tmp/agents.db"

# Initialize Groq client with API key
groq_api_key = "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"
client = groq.Client(api_key=groq_api_key)

# Example chat completion
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Hello! What can you do?"}
    ],
    max_tokens=100
)

web_agent = Agent(
    name="Web Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGoTools()],
    instructions=["Always include sources"],
    # Store the agent sessions in a sqlite database
    storage=SqliteStorage(table_name="web_agent", db_file=agent_storage),
    # Adds the current date and time to the instructions
    add_datetime_to_instructions=True,
    # Adds the history of the conversation to the messages
    add_history_to_messages=True,
    # Number of history responses to add to the messages
    num_history_responses=5,
    # Adds markdown formatting to the messages
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Always use tables to display data"],
    storage=SqliteStorage(table_name="finance_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

app = Playground(agents=[web_agent, finance_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)