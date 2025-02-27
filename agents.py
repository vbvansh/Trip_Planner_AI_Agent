from crewai import Agent
from textwrap import dedent
from langchain.llms import OpenAI, Ollama
from langchain_openai import ChatOpenAI


"""
Creating Agents Cheat Sheet:
- Think like a boss. Work backwards from the goal and think which employee 
    you need to hire to get the job done.
- Define the Captain of the crew who orient the other agents towards the goal. 
- Define which experts the captain needs to communicate with and delegate tasks to.
    Build a top down structure of the crew.

Goal:
- Create a 7-day travel itinerary with detailed per-day plans,
    including budget, packing suggestions, and safety tips.

Captain/Manager/Boss:
- Expert Travel Agent

Employees/Experts to hire:
- City Selection Expert 
- Local Tour Guide


Notes:
- Agents should be results driven and have a clear goal in mind
- Role is their job title
- Goals should actionable
- Backstory should be their resume
"""

class CustomAgents:
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        self.Ollama = Ollama(model="openhermes")

    def expert_travel_agent(self):
        return Agent(
            role="Expert Travel Agent",
            backstory=dedent(f"""Expert in Travel Planning and logistics. I am having decades of experience in creating travel itineraries for clients. I have a strong network of travel agents, tour guides, and local experts to help me create the best travel plans for my clients. I have a deep understanding of different cultures, languages, and travel destinations. I am passionate about creating memorable travel experiences for my clients."""),
            goal=dedent(f"""Create a 7-day travel itinerary with detailed per-day plans, including budget, packing suggestions, and safety tips."""),
            # tools=[tool_1, tool_2],
            # allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def city_selection_expert(self):
        return Agent(
            role="City Selection Expert",
            backstory=dedent(f"""Expert at Analyzing and selecting travel destinations. I have a deep understanding of different cities, their attractions, culture, and safety. I have traveled extensively and have first-hand experience of various cities around the world. I can provide detailed insights into the best cities to visit based on the client's preferences, budget, and interests. I am passionate about helping people discover new and exciting travel destinations."""),
            goal=dedent(f"""Select the best cities to visit based on the client's preferences, budget, and interests."""),
            # tools=[tool_1, tool_2],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )
        
    def local_tour_guide(self):
        return Agent(
            role="Local Tour Guide",
            backstory=dedent(f"""Knowlegable local guide with extensive experience in providing tours to travelers. I have a deep understanding of the local culture, history, and attractions. I can provide personalized tours based on the client's interests, preferences, and budget. I am passionate about sharing my knowledge and love for my city with travelers and creating memorable experiences for them."""),
            goal=dedent(f"""Provide the BEST local tour guide experience for the client."""),
            # tools=[tool_1, tool_2],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )
