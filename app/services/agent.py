from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from firecrawl import FirecrawlApp
import os
import re
import argparse
import json
from dotenv import load_dotenv
import logging
import requests


# Load environment variables
load_dotenv()

class PropertyData(BaseModel):
    """Schema for property data extraction"""
    building_name: str = Field(description="Name of the building/property", alias="Building_name")
    property_type: str = Field(description="Type of property (commercial, residential, etc)", alias="Property_type")
    location_address: str = Field(description="Complete address of the property")
    price: str = Field(description="Price of the property", alias="Price")
    description: str = Field(description="Detailed description of the property", alias="Description")

class PropertiesResponse(BaseModel):
    """Schema for multiple properties response"""
    properties: List[PropertyData] = Field(description="List of property details")

class LocationData(BaseModel):
    """Schema for location price trends"""
    location: str
    price_per_sqft: float
    percent_increase: float
    rental_yield: float

class LocationsResponse(BaseModel):
    """Schema for multiple locations response"""
    locations: List[LocationData] = Field(description="List of location data points")

class FirecrawlResponse(BaseModel):
    """Schema for Firecrawl API response"""
    success: bool
    data: Dict
    status: str
    expiresAt: str

class PropertyFindingAgent:
    """Agent responsible for finding properties and providing recommendations"""
    
    def __init__(self, firecrawl_api_key: str, openai_api_key: str, model_id: str = "gpt-3.5-turbo"):
        self.agent = Agent(
            model=OpenAIChat(id=model_id, api_key=openai_api_key),
            markdown=True,
            description="I am a real estate expert who helps find and analyze properties based on user preferences."
        )
        self.firecrawl = FirecrawlApp(api_key=firecrawl_api_key)
        
        # Create an interpreter agent instead of using OpenAIChat directly
        self.query_interpreter = Agent(
            model=OpenAIChat(id=model_id, api_key=openai_api_key),
            markdown=False,
            description="I extract search parameters from natural language queries."
        )

    def find_properties(
        self, 
        city: str,
        max_price: float,
        property_category: str = "Residential",
        property_type: str = "Flat"
    ) -> str:
        """Find and analyze properties based on user preferences"""
        formatted_location = city.lower()
        
        urls = [
            f"https://www.squareyards.com/sale/property-for-sale-in-{formatted_location}/*",
            f"https://www.99acres.com/property-in-{formatted_location}-ffid/*",
            f"https://housing.com/in/buy/{formatted_location}/{formatted_location}",
            # f"https://www.nobroker.in/property/sale/{city}/{formatted_location}",
        ]
        
        property_type_prompt = "Flats" if property_type == "Flat" else "Individual Houses"
        
        raw_response = self.firecrawl.extract(
            urls=urls,
            params={
                'prompt': f"""Extract ONLY 5 OR LESS different {property_category} {property_type_prompt} from {city} that cost less than {max_price} crores.
                
                Requirements:
                - Property Category: {property_category} properties only
                - Property Type: {property_type_prompt} only
                - Location: {city}
                - Maximum Price: {max_price} crores
                - Include complete property details with exact location
                - IMPORTANT: Return data for at least 3 different properties. MAXIMUM 5.
                - Format as a list of properties with their respective details
                """,
                'schema': PropertiesResponse.model_json_schema()
            }
        )
        
        print("Raw Property Response:", raw_response)
        
        if isinstance(raw_response, dict) and raw_response.get('success'):
            properties = raw_response['data'].get('properties', [])
        else:
            properties = []
            
        print("Processed Properties:", properties)

        
        analysis = self.agent.run(
            f"""As a real estate expert, analyze these properties and market trends:

            Properties Found in json format:
            {properties}

            **IMPORTANT INSTRUCTIONS:**
            1. ONLY analyze properties from the above JSON data that match the user's requirements:
               - Property Category: {property_category}
               - Property Type: {property_type}
               - Maximum Price: {max_price} crores
            2. DO NOT create new categories or property types
            3. From the matching properties, select 5-6 properties with prices closest to {max_price} crores

            Please provide your analysis in this format:
            
            🏠 SELECTED PROPERTIES
            • List only 5-6 best matching properties with prices closest to {max_price} crores
            • For each property include:
              - Name and Location
              - Price (with value analysis)
              - Key Features
              - Pros and Cons

            💰 BEST VALUE ANALYSIS
            • Compare the selected properties based on:
              - Price per sq ft
              - Location advantage
              - Amenities offered

            Format your response in a clear, structured way using the above sections.
            """
        )
        
        return analysis.content

    def get_location_trends(self, city: str) -> str:
        """Get price trends for different localities in the city"""
        raw_response = self.firecrawl.extract([
            f"https://www.99acres.com/property-rates-and-price-trends-in-{city.lower()}-prffid/*"
        ], {
            'prompt': """Extract price trends data for ALL major localities in the city. 
            IMPORTANT: 
            - Return data for at least 5-10 different localities
            - Include both premium and affordable areas
            - Do not skip any locality mentioned in the source
            - Format as a list of locations with their respective data
            """,
            'schema': LocationsResponse.model_json_schema(),
        })
        
        if isinstance(raw_response, dict) and raw_response.get('success'):
            locations = raw_response['data'].get('locations', [])
    
            analysis = self.agent.run(
                f"""As a real estate expert, analyze these location price trends for {city}:

                {locations}

                Please provide:
                1. A bullet-point summary of the price trends for each location
                2. Identify the top 3 locations with:
                   - Highest price appreciation
                   - Best rental yields
                   - Best value for money
                3. Investment recommendations:
                   - Best locations for long-term investment
                   - Best locations for rental income
                   - Areas showing emerging potential
                4. Specific advice for investors based on these trends

                Format the response as follows:
                
                📊 LOCATION TRENDS SUMMARY
                • [Bullet points for each location]

                🏆 TOP PERFORMING AREAS
                • [Bullet points for best areas]

                💡 INVESTMENT INSIGHTS
                • [Bullet points with investment advice]

                🎯 RECOMMENDATIONS
                • [Bullet points with specific recommendations]
                """
            )
            
            return analysis.content
            
        return "No price trends data available"
    
    def interpret_user_query(self, query: str) -> Dict:
        """Use Agent to interpret the user's natural language query"""
        response = self.query_interpreter.run(
            f"""Extract real estate search parameters from the following user query:
            
            User query: "{query}"
            
            Extract these parameters:
            1. City name (required)
            2. Maximum price in crores (default is 5 crores if not specified)
            3. Property category: "Residential" or "Commercial" (default is "Residential" if not specified)
            4. Property type: "Flat" or "Individual House" (default is "Flat" if not specified)
            
            Output in JSON format:
            {{
                "city": "extracted city name",
                "max_price": extracted price as a number or null if not specified,
                "property_category": "extracted category or default",
                "property_type": "extracted type or default"
            }}
            
            Only output valid JSON, no explanations or additional text.
            """
        )
        
        try:
            # Try to extract JSON from the response
            match = re.search(r'({.*})', response.content, re.DOTALL)
            if match:
                params = json.loads(match.group(1))
                return params
            else:
                # If no JSON found, try to manually extract parameters
                city_match = re.search(r'city["\s:]+([^",\s]+)', response.content, re.IGNORECASE)
                city = city_match.group(1) if city_match else None
                
                return {
                    "city": city,
                    "max_price": 5.0,  # Default
                    "property_category": "Residential",  # Default
                    "property_type": "Flat"  # Default
                }
        except Exception as e:
            print(f"Error parsing interpreter response: {e}")
            print(f"Raw response: {response.content}")
            return {}

# def main():
#     # Set up command-line argument parsing
#     parser = argparse.ArgumentParser(description="AI Real Estate Agent")
#     parser.add_argument("query", nargs="*", help="Natural language query, e.g., 'best properties in Delhi'")
#     args = parser.parse_args()
    
#     # Get API keys from environment variables
#     firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
#     openai_key = os.getenv("OPENAI_API_KEY")
#     model_id = os.getenv("OPENAI_MODEL_ID", "gpt-3.5-turbo")
    
#     if not firecrawl_key or not openai_key:
#         print("❌ Error: Missing API keys. Please set FIRECRAWL_API_KEY and OPENAI_API_KEY in your .env file.")
#         return
    
#     # Create the property agent
#     property_agent = PropertyFindingAgent(
#         firecrawl_api_key=firecrawl_key,
#         openai_api_key=openai_key,
#         model_id=model_id
#     )
    
#     # Get user query
#     query = " ".join(args.query) if args.query else input("What properties are you looking for? (e.g., 'best properties in Delhi'): ")
    
#     if not query:
#         print("❌ Please provide a search query!")
#         return
    
#     try:
#         # Interpret the user's query
#         print("🔍 Analyzing your query...")
#         params = property_agent.interpret_user_query(query)
        
#         if not params.get("city"):
#             print("❌ Couldn't determine which city you're interested in. Please specify a city.")
#             return
        
#         city = params.get("city")
#         max_price = params.get("max_price", 5.0)
#         property_category = params.get("property_category", "Residential")
#         property_type = params.get("property_type", "Flat")
        
#         print(f"\n📋 Search Parameters:")
#         print(f"City: {city}")
#         print(f"Maximum Price: {max_price} crores")
#         print(f"Property Category: {property_category}")
#         print(f"Property Type: {property_type}")
#         print("\n" + "-" * 50 + "\n")
        
#         # Search for properties
#         print(f"🔍 Searching for properties in {city}...")
#         property_results = property_agent.find_properties(
#             city=city,
#             max_price=max_price,
#             property_category=property_category,
#             property_type=property_type
#         )
        
#         print("\n🏘️ PROPERTY RECOMMENDATIONS")
#         print("-" * 50)
#         print(property_results)
#         print("-" * 50)
        
#         # Get location trends
#         print(f"\n📊 Analyzing location trends in {city}...")
#         location_trends = property_agent.get_location_trends(city)
        
#         print("\n📈 LOCATION TRENDS ANALYSIS")
#         print("-" * 50)
#         print(location_trends)
#         print("-" * 50)
        
#     except Exception as e:
#         print(f"❌ An error occurred: {str(e)}")


def generate_response(message_body):
            """Generates a response for property search queries."""
            try:            
                # Get API keys from environment variables
                firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
                openai_key = os.getenv("OPENAI_API_KEY")
                model_id = os.getenv("OPENAI_MODEL_ID", "gpt-3.5-turbo")

                if not firecrawl_key or not openai_key:
                            return "❌ Error: Missing API keys. Please set FIRECRAWL_API_KEY and OPENAI_API_KEY in your .env file."
        
                # Create the property agent
                property_agent = PropertyFindingAgent(
                    firecrawl_api_key=firecrawl_key,
                    openai_api_key=openai_key,
                    model_id=model_id
                )
        
                if not message_body:
                            return "❌ Please provide a search query!"
        
                # Interpret the user's query
                params = property_agent.interpret_user_query(message_body)
        
                if not params.get("city"):
                            return "❌ Couldn't determine which city you're interested in. Please specify a city."
        
                city = params.get("city")
                max_price = params.get("max_price", 5.0)
                property_category = params.get("property_category", "Residential")
                property_type = params.get("property_type", "Flat")
        
                # Search for properties
                property_results = property_agent.find_properties(
                    city=city,
                    max_price=max_price,
                    property_category=property_category,
                    property_type=property_type
                )
        
                # Get location trends
                
                
                # location_trends = property_agent.get_location_trends(city)
        
                # Format the response
                response = (
                    # f"\n📋 Search Parameters:\n"
                    # f"City: {city}\n"
                    # f"Maximum Price: {max_price} crores\n"
                    # f"Property Category: {property_category}\n"
                    # f"Property Type: {property_type}\n"
                    
                    f"\n🏘️ PROPERTY RECOMMENDATIONS\n"
                    "--------------------------------------------------\n"
                    f"{property_results}\n"
                    
                   
                )
                return response
            except Exception as e:
                logging.error(f"Error generating response: {e}")
                return "❌ An error occurred while processing your request. Please try again later."



# implementation of thread is not done yet , will do i f needed

if __name__ == "__main__":
    while True:
        user_input = input("Enter your query (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        response = generate_response(user_input)
        print("\nResponse:\n", response)