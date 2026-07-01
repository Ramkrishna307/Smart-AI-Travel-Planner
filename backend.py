import os
import requests
from dotenv import load_dotenv
import certifi
os.environ["SSL_CERT_FILE"]=certifi.where()
os.environ["REQUESTS_CA_BUNDLE"]=certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid 

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START,END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)


from langchain_groq import ChatGroq
from tools.travily import tavily_search
from tools.flight import search_flights


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




class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int



#Flight Search Agent:
def flight_search_agent(state: TravelState):
    query = state['user_query']
    flight_data=search_flights(query)
    
    return {
        "flight_results":flight_data,
        "messages": [
            AIMessage(content=f"Flight Search fethed results.")

        ],
        "llm_calls": state.get("llm_calls",0)+1
    }

def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# Itineary Generation agent

def itinearary_agent(state: TravelState):
    prompt=f"""
    Create a complete travel itinerary for 

    User Query:
    {state['user_query']}
    Flight Results:
    {state['flight_results']}
    Hotel Results:
    {state['hotel_results']}

    Make the itinerary practical, budget-aware, and easy to follow.

  """
    
    response=llm.invoke(
        [
            SystemMessage(content="You are a expert travel planner"),
            HumanMessage(content=prompt)
        ]
    )

    return {
        "itinerary": response.content,
        "messages":[response],
        "llm_calls": state.get("llm_calls", 0)+1
    }
    
#Final response agent

def final_response_agent(state: TravelState):
    final_prompt=f"""
Generate the Final response for the user based on the following information:
User request:
{state["user_query"]}

Flights:
{state["flight_results"]}

Hotels:
{state["hotel_results"]}

Ittinerary:
{state["itinerary"]}

format the final answer in a user-friendly manner, making it easy to read and follow. Use bullet points, headings, and any other formatting that enhances clarity. Ensure that the response is concise yet comprehensive, providing all necessary details for the user's travel plans.

1.Trip Summary: Provide a brief overview of the trip, including key highlights and essential information.
2. Flight Details: Summarize the flight information, including departure and arrival times, airlines, and any layovers.
3. Hotel Information: Present the hotel details, including names, addresses, check-in/check-out
4. Day-by-Day Itinerary: Break down the itinerary into daily plans, including activities, sightseeing, and any special events or recommendations.
5. Extimated Budget: Provide an estimated budget for the trip, including flights, accommodation, meals, and activities.
6. Final Recommendations: Offer any additional tips, recommendations, or important notes for the traveler to ensure a smooth and enjoyable experience.

Improtant
- Be clear and practical
- Mention that live flight API may not provde ticket prices if pricing is unavailable
-keep the response useful for real travel planning

 """
    response = llm.invoke([
    SystemMessage(content="You are a professional AI travel booking assistant."),
    HumanMessage(content=final_prompt)
    ])
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


graph=StateGraph(TravelState)

graph.add_node("flight_agent", flight_search_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinearary_agent)
graph.add_node("Final_response_agent", final_response_agent)


graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent","itinerary_agent" )
graph.add_edge("itinerary_agent", "Final_response_agent")
graph.add_edge("Final_response_agent", END)



# database Connection
#PostfreSQL Checkpointer

DATABASE_URL=get_database_url()
_conn=psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row   
)

checkpointer=PostgresSaver(_conn)
checkpointer.setup()

travel_graph=graph.compile(checkpointer=checkpointer)



# FastApI config

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if thread_id is None:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0,
        },
        config=config,
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }