from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()

def get_search_tool(max_results: int = 6):
    return TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))