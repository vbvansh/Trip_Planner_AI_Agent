# To know more about the Task class, visit: https://docs.crewai.com/concepts/tasks
from crewai import Task
from textwrap import dedent


class TravelTasks:
    def __tip_section(self):
        return "If you do your BEST WORK, I'll give you a $10,000 commission!"

    def plan_itinerary(self, agent, city, travel_dates,interests):
        return Task(
            description=dedent(
                f"""
          **Task**: Develop a 7-day Travel Itinerary
       **Description**: Expand the city guide into a full 7-day travel itinerary with detailed 
                per-day plans, including weather forecasts, places to eat, packing suggestions, 
                and a budget breakdown. You MUST suggest actual places to visit, actual hotels to stay, 
                and actual restaurants to go to. This itinerary should cover all aspects of the trip, 
                from arrival to departure, integrating the city guide information with practical travel logistics.
       **Parameters**: 
        - City : {city}
        - Trip Date : {travel_dates}
        - Traveler Interests : {interests}

         **Note**: {self.__tip_section()}
        """
            ),
         
            agent=agent,
            expected_output="A detailed 7-day travel itinerary with all specified details"
        )

    def identify_city(self, agent, origin, cities, interests, travel_dates):
        return Task(
            description=dedent(
                f"""
       **Task**: Identifyy the Best City to Visit
       **Description**: Based on the traveler's origin, interests, and travel dates, recommend the best city to visit. 
                Consider the traveler's interests, the weather, and the availability of activities in the city. 
                Provide a detailed explanation of why this city is the best choice for the traveler.
                Your final answer must be a detailed report on the chosen city, 
                including actual flight costs, weather forecast, and attractions.
       **Parameters**: 
       - Origin : {origin}
       - Cities : {cities}
       - Interests : {interests}
       - Travel Dates : {travel_dates}
    

         **Note**: {self.__tip_section()}
        """
            ),
            expected_output="The expected output of the task",
            agent=agent,
        )

    def gather_city_info(self, agent, city, travel_dates, interests):
        return Task (
            description=dedent(
                f"""
                    **Task**:  Gather In-depth City Guide Information
                    **Description**: Compile an in-depth guide for the selected city, gathering information about 
                        key attractions, local customs, special events, and daily activity recommendations. 
                        This guide should provide a thorough overview of what the city has to offer, including 
                        hidden gems, cultural hotspots, must-visit landmarks, weather forecasts, and high-level costs.

                    **Parameters**: 
                    - Cities: {city}
                    - Interests: {interests}
                    - Travel Date: {travel_dates}

                    **Note**: {self.__tip_section()}
        """
            ),
            expected_output="The expected output of the task",
            agent=agent,
)