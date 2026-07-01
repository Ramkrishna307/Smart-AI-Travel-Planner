from tools.travily import tavily_search
from backend import run_travel_agent
# res=print(tavily_search("Best hotel in India"))
# print(res)

res=run_travel_agent("Best travel plan for 5 days in India with budget of 1000 USD")
print(res)