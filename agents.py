from crewai import Agent
from textwrap import dedent
from tools.search_tools import SearchTools
from tools.calculator_tools import CalculatorTools
import os
from dotenv import load_dotenv
import litellm

# Load environment variables
load_dotenv()

class TravelAgents:
    def __init__(self):
        # Get API key from environment variables
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")

        # ✅ Set the API key for LiteLLM
        os.environ["GROQ_API_KEY"] = groq_api_key  

        # ✅ Define the LLM model
        self.llm = lambda messages: litellm.completion(
            model="groq/llama3-8b-8192",
            messages=messages
        )

    def expert_travel_agent(self):
        return Agent(
            role="Expert Travel Agent",
            backstory=dedent("""Expert in Travel Planning and logistics..."""),  
            goal=dedent("""Create a 7-day travel itinerary..."""),  
            tools=[SearchTools.search_internet, CalculatorTools.calculate],  
            verbose=True,  
            llm=self.llm,  # ✅ Pass the function, not a response object
        )

    def city_selection_expert(self):
        return Agent(
            role="City Selection Expert",
            backstory=dedent("""Expert at analyzing and selecting travel destinations..."""),  
            goal=dedent("""Select the best cities to visit..."""),  
            tools=[SearchTools.search_internet],  
            allow_delegation=False,  
            verbose=True,  
            llm=self.llm,  # ✅ Pass the function, not a response object
        )

    def local_tour_guide(self):
        return Agent(
            role="Local Tour Guide",
            backstory=dedent("""Knowledgeable local guide with extensive experience..."""),  
            goal=dedent("""Provide the BEST local tour guide experience..."""),  
            tools=[SearchTools.search_internet],  
            allow_delegation=False,  
            verbose=True,  
            llm=self.llm,  # ✅ Pass the function, not a response object
        )
