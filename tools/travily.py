from tavily import TavilyClient
import os
from dotenv import load_dotenv


load_dotenv()

client=TavilyClient(
    api_key=os.getenv("TRAVILY_API_KEY")
)

def tavily_search(query):
    reponse=client.search(
        query=query,
        max_results=5
    )
    results=[]

    for i, r in enumerate(reponse["results"], 1):
        title=r.get("title","Unknown")
        url=r.get("url","")
        snippet=r.get("content","").strip()
        #keep only the first 300 characters to avooid wal-of text
        if len(snippet)>300:
            snippet=snippet[:300].rsplit(" ",1)[0]+ "..."
        
        results.append(f"{i}.**{title}**\n {url}\n {snippet}")
    
    return "\n\n".join(results)