from dotenv import load_dotenv
import os
load_dotenv()

import json
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from tools import get_search_tool

class ResearchState(TypedDict):
    topic: str
    search_results: str
    summaries: str
    fact_check: str
    final_report: str
    current_agent: str
    logs: List[str]

def get_llm(temperature: float = 0.3):
    api_key = os.getenv("GROQ_API_KEY")
    print(f"🔑 Groq key loaded: {bool(api_key)}")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=temperature,
        api_key=api_key
    )
# ─── Agent 1: Web Searcher ────────────────────────────────────────────────────

def web_searcher_agent(state: ResearchState) -> ResearchState:
    topic = state["topic"]
    logs = state.get("logs", [])
    logs.append("🔍 Web Searcher Agent activated — searching the web...")

    try:
        client = get_search_tool()
        response = client.search(topic, max_results=6)
        results = response.get("results", [])
        
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"SOURCE {i}:\nURL: {r.get('url', 'N/A')}\nTitle: {r.get('title', 'N/A')}\nContent: {r.get('content', '')[:800]}"
            )
        search_text = "\n\n---\n\n".join(formatted)
        logs.append(f"✅ Found {len(results)} sources from the web.")
    except Exception as e:
        search_text = f"Search failed: {str(e)}"
        logs.append(f"❌ Search error: {str(e)}")

    return {
        **state,
        "search_results": search_text,
        "current_agent": "summarizer",
        "logs": logs
    }
# ─── Agent 2: Paper Summarizer ────────────────────────────────────────────────

def summarizer_agent(state: ResearchState) -> ResearchState:
    logs = state.get("logs", [])
    logs.append("📄 Summarizer Agent activated — extracting key insights...")

    llm = get_llm(temperature=0.2)
    messages = [
        SystemMessage(content="""Summarize the key findings from these search results in bullet points.
Cover: main facts, important data, key themes. Be concise."""),
        HumanMessage(content=f"""Topic: {state['topic']}

Raw Search Results:
{state['search_results']}

Provide a structured summary of the key information found.""")
    ]

    response = llm.invoke(messages)
    summaries = response.content
    logs.append("✅ Summarization complete — key insights extracted.")

    return {
        **state,
        "summaries": summaries,
        "current_agent": "fact_checker",
        "logs": logs
    }

# ─── Agent 3: Fact Checker ────────────────────────────────────────────────────

def fact_checker_agent(state: ResearchState) -> ResearchState:
    logs = state.get("logs", [])
    logs.append("🔎 Fact-Checker Agent activated — verifying claims...")

    llm = get_llm(temperature=0.1)
    messages = [
       SystemMessage(content="""Fact-check these findings briefly.
List: VERIFIED CLAIMS, UNCERTAIN CLAIMS, RELIABILITY RATING (High/Medium/Low)."""),
        HumanMessage(content=f"""Topic: {state['topic']}

Summaries to fact-check:
{state['summaries']}

Original sources:
{state['search_results'][:3000]}

Perform fact-checking analysis.""")
    ]

    response = llm.invoke(messages)
    fact_check = response.content
    logs.append("✅ Fact-checking complete — reliability assessed.")

    return {
        **state,
        "fact_check": fact_check,
        "current_agent": "report_writer",
        "logs": logs
    }

# ─── Agent 4: Report Writer ───────────────────────────────────────────────────

def report_writer_agent(state: ResearchState) -> ResearchState:
    logs = state.get("logs", [])
    logs.append("✍️ Report Writer Agent activated — composing final report...")

    llm = get_llm(temperature=0.4)
    messages = [
        SystemMessage(content="""Write a research report with these sections:
## Summary
## Key Findings  
## Analysis
## Conclusion
Keep it focused and under 500 words."""),
        HumanMessage(content=f"""Topic: {state['topic']}

Research Summaries:
{state['summaries']}

Fact-Check Analysis:
{state['fact_check']}

Write a complete, professional research report.""")
    ]

    response = llm.invoke(messages)
    report = response.content
    logs.append("✅ Final report generated successfully!")

    return {
        **state,
        "final_report": report,
        "current_agent": "done",
        "logs": logs
    }

# ─── Build LangGraph ──────────────────────────────────────────────────────────

def build_research_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("web_searcher", web_searcher_agent)
    graph.add_node("summarizer", summarizer_agent)
    graph.add_node("fact_checker", fact_checker_agent)
    graph.add_node("report_writer", report_writer_agent)

    graph.set_entry_point("web_searcher")
    graph.add_edge("web_searcher", "summarizer")
    graph.add_edge("summarizer", "fact_checker")
    graph.add_edge("fact_checker", "report_writer")
    graph.add_edge("report_writer", END)

    return graph.compile()

research_graph = build_research_graph()