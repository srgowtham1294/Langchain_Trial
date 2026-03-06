import os
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

google_api_key = os.getenv('GEMINI_API_KEY')

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet
    Args:
        query (str): the query to search
    Returns:
        str: the search result
    """
    print('Searching for "' + query + '"')
    return tavily.search(query=query)

model_1 = 'llama-3.3-70b-versatile'
model_2 = 'meta-llama/llama-4-maverick-17b-128e-instruct'

#llm = ChatGroq(model=model_2)  # ✅ tool-use model
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash',google_api_key=google_api_key)

tools = [search]
# tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-search-agent-starter!")
    result = agent.invoke({"messages": [HumanMessage(content="Search for 3 job postings for an AI Engineer using langchain in Chennai on linkedin and list their details")]})  # ✅ list
    print(result['messages'][-1].content)  # ✅ print just the final response

if __name__ == "__main__":
    main()

# from groq import Groq
# import os
#
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# models = client.models.list()
# for model in models.data:
#     print(model.id)
