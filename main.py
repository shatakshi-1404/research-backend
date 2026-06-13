from dotenv import load_dotenv
import os
load_dotenv()

print("🔑 GROQ_API_KEY loaded:", bool(os.getenv("GROQ_API_KEY")))
print("🔑 TAVILY_API_KEY loaded:", bool(os.getenv("TAVILY_API_KEY")))
import json
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agents import research_graph, ResearchState

app = FastAPI(title="Multi-Agent Research API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    topic: str

@app.post("/research/stream")
async def research_stream(request: ResearchRequest):
    async def event_generator():
        initial_state: ResearchState = {
            "topic": request.topic,
            "search_results": "",
            "summaries": "",
            "fact_check": "",
            "final_report": "",
            "current_agent": "web_searcher",
            "logs": []
        }

        sent_logs = set()

        try:
            for step in research_graph.stream(initial_state, stream_mode="updates"):
                for node_name, node_state in step.items():
                    yield {
                        "event": "agent_update",
                        "data": json.dumps({
                            "agent": node_name,
                            "status": "running"
                        })
                    }
                    await asyncio.sleep(0.1)

                    new_logs = node_state.get("logs", [])
                    for log in new_logs:
                        if log not in sent_logs:
                            sent_logs.add(log)
                            yield {
                                "event": "log",
                                "data": json.dumps({"message": log})
                            }
                            await asyncio.sleep(0.05)

                    if node_name == "web_searcher" and node_state.get("search_results"):
                        yield {
                            "event": "search_results",
                            "data": json.dumps({
                                "content": node_state["search_results"]
                            })
                        }
                    elif node_name == "summarizer" and node_state.get("summaries"):
                        yield {
                            "event": "summaries",
                            "data": json.dumps({
                                "content": node_state["summaries"]
                            })
                        }
                    elif node_name == "fact_checker" and node_state.get("fact_check"):
                        yield {
                            "event": "fact_check",
                            "data": json.dumps({
                                "content": node_state["fact_check"]
                            })
                        }
                    elif node_name == "report_writer" and node_state.get("final_report"):
                        yield {
                            "event": "final_report",
                            "data": json.dumps({
                                "content": node_state["final_report"]
                            })
                        }

                    yield {
                        "event": "agent_update",
                        "data": json.dumps({
                            "agent": node_name,
                            "status": "done"
                        })
                    }
                    await asyncio.sleep(0.1)

        except Exception as e:
            yield {
                "event": "log",
                "data": json.dumps({"message": f"❌ Server error: {str(e)}"})
            }

        yield {
            "event": "complete",
            "data": json.dumps({"message": "Research complete!"})
        }

    return EventSourceResponse(event_generator())

@app.get("/health")
def health():
    return {"status": "ok"}