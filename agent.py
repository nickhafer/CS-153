import os
import json
import aiohttp
import asyncio
from mistralai import Mistral
import discord
import logging
import time
import re
from collections import defaultdict

MISTRAL_MODEL = "mistral-large-latest"

# Setup logging
logger = logging.getLogger("discord.agent")
logger.setLevel(logging.INFO)

SYSTEM_PROMPT = """
You are a helpful assistant that provides information about recreational activities and 
facilities in the United States. You can help users find camping spots, hiking trails, 
fishing locations, and other outdoor activities.

When providing information about recreational facilities, you MUST:
1. List ALL available results (up to 10 locations)
2. For each facility/area include:
   - The name of the facility
   - A brief description
   - Location details
   - Any relevant amenities or features
   - Links to more information when available

Be concise but informative, and always provide multiple options when available. Never limit your response to just one location unless only one result was found.
"""

EXTRACT_LOCATION_PROMPT = """
Is this message explicitly requesting recreation information for a specific city/location?
If not, return {"location": "none"}.

Otherwise, return the full name of the city in JSON format.

Examples:
Message: Where can I fish in Salt Lake City?
Response: {"location": "Salt Lake City, UT"}

Message: What are the closest campgrounds to Bozeman?
Response: {"location": "Bozeman, MT"}

Message: Are there hiking trails near Boston?
Response: {"location": "Boston, MA"}

Message: Give me the hiking trails in Boulder.
Response: {"location": "Boulder, CO"}

Message: I love hiking in sf!
Response: {"location": "none"}

Message: Is camping fun in NYC?
Response: {"location": "none"}
"""

EXTRACT_ACTIVITY_PROMPT = """
Is this message explicitly requesting recreation information for a specific activity?
If not, return {"ActivityName": "none"}.

Otherwise, return the activity name in JSON format using one of these valid activity types:
BIKING, CLIMBING, CAMPING, FISHING, HIKING, HUNTING, WINTER SPORTS, WATER SPORTS, RECREATIONAL VEHICLES, WILDLIFE VIEWING, OTHER

Example:
Message: Where can I fish in Salt Lake City?
Response: {"ActivityName": "FISHING"}

Message: What are the closest campgrounds to Bozeman?
Response: {"ActivityName": "CAMPING"}

Message: Are there hiking trails near Boston?
Response: {"ActivityName": "HIKING"}

Message: Give me the hiking trails in Boulder.
Response: {"ActivityName": "HIKING"}

Message: I love hiking in sf!
Response: {"ActivityName": "none"}

Message: Is camping fun in NYC?
Response: {"ActivityName": "none"}
"""

EXTRACT_RADIUS_PROMPT = """
Is this message explicitly requesting recreation information for a specific radius?
If not, return {"Radius": "25"}.

Otherwise, return the radius specified. If the value is greater than 50, return 50.

Examples:
Message: Where can I fish within 30 miles of Salt Lake City?
Response: {"Radius": "30"}

Message: What are the closest campgrounds to Bozeman?
Response: {"Radius": "10"}

Message: Are there hiking trails within 60 miles of Boston?
Response: {"Radius": "50"}

Message: Give me the hiking trails in Boulder.
Response: {"Radius": "10"}

Message: I love hiking in sf!
Response: {"Radius": "10"}

Message: Is camping fun in NYC?
Response: {"Radius": "10"}
"""

EXTRACT_LIMIT_PROMPT = """
Is this message explicitly requesting a limit on the number of results to return?
If not, return {"Limit": "5"}.

Otherwise, return the limit specified, with a maximum of 10. 

Examples:
Message: Show me the 3 best places to fish near Salt Lake City.
Response: {"Limit": "3"}

Message: What are the closest campgrounds to Bozeman?
Response: {"Limit": "5"}

Message: Show me 11 hiking trails within 60 miles of Boston?
Response: {"Limit": "10"}

Message: Give me the hiking trails in Boulder.
Response: {"Limit": "5"}
"""

SUMMARIZE_RESULTS_PROMPT = """
You are a helpful assistant that provides information about recreational activities and facilities in the United States.

Below are the results from a search for recreational facilities. You MUST summarize ALL of these results in a friendly, informative way.
For EACH facility/area, include:
- Name and type of facility
- Brief description
- Location
- Key amenities or features
- Any special notes

Format the information in a clear, readable way using numbered lists or bullet points to separate each location.
If there are no results, politely inform the user and suggest they try a different search.

Remember: You must include ALL results provided (up to 10 locations). Do not limit your response to just one location unless only one result was found.

Search query: {query}
Search results: {results}
"""


class MistralAgent:
    def __init__(self):
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        self.RIDB_API_KEY = os.getenv("RIDB_API_KEY")
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.base_url = "https://ridb.recreation.gov/api/v1"
        self.headers = {
            "apikey": self.RIDB_API_KEY,
            "accept": "application/json"
        }
        # Add rate limiting variables
        self.last_request_time = 0
        self.request_interval = 1.0  # Minimum time between requests in seconds
        
        # Add conversation memory
        self.conversation_history = defaultdict(list)
        self.max_history_length = 5  # Keep last 5 interactions per user

    async def extract_parameter(self, message_content, prompt):
        """Extract parameters from user message using Mistral with rate limiting"""
        # Implement rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_interval:
            await asyncio.sleep(self.request_interval - time_since_last_request)
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": message_content},
        ]

        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.last_request_time = time.time()
                response = await self.client.chat.complete_async(
                    model=MISTRAL_MODEL,
                    messages=messages,
                )
                
                result = response.choices[0].message.content
                # Clean up the result to handle code blocks
                if "```json" in result:
                    result = result.replace("```json", "").replace("```", "").strip()
                
                try:
                    # Extract just the JSON part from the response
                    # Look for the first occurrence of a JSON-like pattern
                    json_pattern = r'(\{.*?\})'
                    json_match = re.search(json_pattern, result, re.DOTALL)
                    
                    if json_match:
                        json_str = json_match.group(1)
                        return json.loads(json_str)
                    else:
                        logger.error(f"No JSON pattern found in Mistral response: {result}")
                        # Return a default value based on the prompt type
                        if "location" in prompt:
                            return {"location": "none"}
                        elif "ActivityName" in prompt:
                            return {"ActivityName": "none"}
                        elif "Radius" in prompt:
                            return {"Radius": "25"}
                        elif "Limit" in prompt:
                            return {"Limit": "5"}
                        return {}
                    
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from Mistral response: {result}")
                    # Return a default value based on the prompt type
                    if "location" in prompt:
                        return {"location": "none"}
                    elif "ActivityName" in prompt:
                        return {"ActivityName": "none"}
                    elif "Radius" in prompt:
                        return {"Radius": "25"}
                    elif "Limit" in prompt:
                        return {"Limit": "5"}
                    return {}
                    
            except Exception as e:
                retry_count += 1
                logger.warning(f"API request failed (attempt {retry_count}/{max_retries}): {str(e)}")
                if "429" in str(e):  # Rate limit error
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.info(f"Rate limited. Waiting {wait_time} seconds before retry.")
                    await asyncio.sleep(wait_time)
                else:
                    # For other errors, wait a short time
                    await asyncio.sleep(1)
                
                # If we've reached max retries, return default values
                if retry_count >= max_retries:
                    logger.error(f"Max retries reached. Returning default value.")
                    if "location" in prompt:
                        return {"location": "none"}
                    elif "ActivityName" in prompt:
                        return {"ActivityName": "none"}
                    elif "Radius" in prompt:
                        return {"Radius": "25"}
                    elif "Limit" in prompt:
                        return {"Limit": "5"}
                    return {}

    async def get_coordinates(self, location):
        """Get latitude and longitude for a location using a geocoding service"""
        if location == "none":
            return None, None
            
        async with aiohttp.ClientSession() as session:
            # Using OpenStreetMap Nominatim API for geocoding
            url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1"
            headers = {"User-Agent": "DiscordRecreationBot/1.0"}
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        return data[0]["lat"], data[0]["lon"]
                    else:
                        logger.warning(f"No coordinates found for location: {location}")
                        return None, None
                else:
                    logger.error(f"Error getting coordinates: {response.status}")
                    return None, None

    async def search_facilities(self, lat, lon, activity, radius, limit):
        """Search for recreational facilities using the RIDB API"""
        if not lat or not lon:
            return []
            
        params = {
            "latitude": lat,
            "longitude": lon,
            "radius": radius,
            "limit": limit,  # Use the provided limit directly
            "full": "true"
        }
        
        if activity and activity != "none":
            params["activity"] = activity
            
        logger.info(f"Searching facilities with params: {params}")
            
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/facilities"
            async with session.get(url, headers=self.headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("RECDATA", [])
                else:
                    logger.error(f"Error searching facilities: {response.status}")
                    error_text = await response.text()
                    logger.error(f"Error details: {error_text}")
                    return []

    async def search_recreation_areas(self, lat, lon, activity, radius, limit):
        """Search for recreational areas using the RIDB API"""
        if not lat or not lon:
            return []
            
        params = {
            "latitude": lat,
            "longitude": lon,
            "radius": radius,
            "limit": limit,  # Use the provided limit directly
            "full": "true"
        }
        
        if activity and activity != "none":
            params["activity"] = activity
            
        logger.info(f"Searching recreation areas with params: {params}")
            
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/recareas"
            async with session.get(url, headers=self.headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("RECDATA", [])
                else:
                    logger.error(f"Error searching recreation areas: {response.status}")
                    return []

    async def summarize_results(self, query, results):
        """Use Mistral to summarize the search results in a user-friendly way"""
        if not results:
            return "I couldn't find any recreational facilities matching your criteria. Please try a different search. You could try expanding the radius or changing the activity or specifying a different location."
            
        # Don't limit to just 5 results anymore
        results_str = json.dumps(results, indent=2)
        if len(results_str) > 4000:
            results_str = results_str[:4000] + "... (truncated)"
            
        prompt = SUMMARIZE_RESULTS_PROMPT.format(query=query, results=results_str)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )
        
        return response.choices[0].message.content

    async def run(self, message: discord.Message):
        """Process a user message and return a response"""
        user_query = message.content
        user_id = str(message.author.id)
        
        try:
            # Check conversation history for failed searches
            previous_failed_search = False
            previous_location = None
            previous_activity = None
            
            if self.conversation_history[user_id]:
                last_interaction = self.conversation_history[user_id][-1]
                if "no_results" in last_interaction and last_interaction["no_results"]:
                    previous_failed_search = True
                    previous_location = last_interaction.get("location")
                    previous_activity = last_interaction.get("activity")
                    
                    # Check if this is a request to expand search with a custom radius
                    expand_radius_match = re.search(r'expand\s+(?:search|radius)\s+(?:to|by)?\s*(\d+)', user_query.lower())
                    if expand_radius_match:
                        custom_radius = int(expand_radius_match.group(1))
                        # Cap at 50 miles
                        custom_radius = min(custom_radius, 50)
                        user_query = f"Find {previous_activity if previous_activity != 'none' else 'recreational activities'} near {previous_location} within {custom_radius} miles"
                        logger.info(f"Expanding previous search with custom radius: {user_query}")
                    # Check if this is a simple request to expand search
                    elif "expand" in user_query.lower() or "try again" in user_query.lower() or "search again" in user_query.lower():
                        user_query = f"Find {previous_activity if previous_activity != 'none' else 'recreational activities'} near {previous_location} with expanded radius"
                        logger.info(f"Expanding previous search: {user_query}")
            
            # Extract parameters from the user's message with delay between requests
            location_data = await self.extract_parameter(user_query, EXTRACT_LOCATION_PROMPT)
            await asyncio.sleep(1.5)  # Add delay between API calls
            
            activity_data = await self.extract_parameter(user_query, EXTRACT_ACTIVITY_PROMPT)
            await asyncio.sleep(1.5)  # Add delay between API calls
            
            radius_data = await self.extract_parameter(user_query, EXTRACT_RADIUS_PROMPT)
            await asyncio.sleep(1.5)  # Add delay between API calls
            
            limit_data = await self.extract_parameter(user_query, EXTRACT_LIMIT_PROMPT)
            
            location = location_data.get("location", "none")
            activity = activity_data.get("ActivityName", "none")
            radius = radius_data.get("Radius", "25")
            limit = limit_data.get("Limit", "5")
            
            # If this is a follow-up to a failed search and no new location specified, use the previous one
            if previous_failed_search and location == "none" and previous_location:
                location = previous_location
                logger.info(f"Using previous location: {location}")
                
            # If this is a follow-up to a failed search and no new activity specified, use the previous one
            if previous_failed_search and activity == "none" and previous_activity:
                activity = previous_activity
                logger.info(f"Using previous activity: {activity}")
            
            # Convert to integers with defaults
            try:
                radius = int(radius) if radius != "none" else 25
                limit = int(limit) if limit != "none" else 5
            except ValueError:
                radius = 25
                limit = 5
                
            # If this is a follow-up to a failed search, start with a larger radius
            if previous_failed_search and "expand" in user_query.lower():
                # Check if a custom radius was specified in the expand command
                expand_radius_match = re.search(r'expand\s+(?:search|radius)\s+(?:to|by)?\s*(\d+)', user_query.lower())
                if expand_radius_match:
                    custom_radius = int(expand_radius_match.group(1))
                    radius = min(custom_radius, 50)  # Cap at 50 miles
                    logger.info(f"Using custom expansion radius: {radius} miles")
                else:
                    # Default expansion - at least 30 miles
                    radius = max(radius, 30)
                    logger.info(f"Using default expansion radius: {radius} miles")
            
            # Cap values
            radius = min(radius, 50)
            limit = min(limit, 10)
            
            # Ensure we get at least 3 results by default
            min_results = 3
            
            # If no specific location or activity was detected, use Mistral to generate a general response
            if location == "none":
                await asyncio.sleep(1.5)  # Add delay before API call
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                ]
                self.last_request_time = time.time()
                response = await self.client.chat.complete_async(
                    model=MISTRAL_MODEL,
                    messages=messages,
                )
                response_text = response.choices[0].message.content
                # Truncate if longer than Discord's limit
                if len(response_text) > 1990:
                    response_text = response_text[:1990] + "..."
                    
                # Store in conversation history
                self.conversation_history[user_id].append({
                    "query": user_query,
                    "response": response_text,
                    "no_results": False
                })
                
                # Limit history size
                if len(self.conversation_history[user_id]) > self.max_history_length:
                    self.conversation_history[user_id].pop(0)
                    
                return response_text
                
            # Get coordinates for the location
            lat, lon = await self.get_coordinates(location)
            
            if not lat or not lon:
                response_text = f"I couldn't find the coordinates for '{location}'. Please try a different location or be more specific."
                
                # Store in conversation history
                self.conversation_history[user_id].append({
                    "query": user_query,
                    "response": response_text,
                    "no_results": True,
                    "location": location,
                    "activity": activity
                })
                
                # Limit history size
                if len(self.conversation_history[user_id]) > self.max_history_length:
                    self.conversation_history[user_id].pop(0)
                    
                return response_text
            
            # Log the search parameters for debugging
            logger.info(f"Searching for {activity} near {location} within {radius} miles with limit {limit}")
                
            # Search for facilities and recreation areas with initial radius
            # Use a higher limit to ensure we get enough results
            api_limit = max(limit * 3, 10)  # At least 10 results from each API
            
            facilities = await self.search_facilities(lat, lon, activity, radius, api_limit)
            rec_areas = await self.search_recreation_areas(lat, lon, activity, radius, api_limit)
            
            # Log the number of results found
            logger.info(f"Found {len(facilities)} facilities and {len(rec_areas)} recreation areas")
            
            # Combine results
            combined_results = []
            
            # Add facilities with source
            for facility in facilities:
                facility['source'] = 'facility'
                combined_results.append(facility)
                
            # Add rec areas with source
            for rec_area in rec_areas:
                rec_area['source'] = 'rec_area'
                combined_results.append(rec_area)
            
            # If we don't have enough results, try with an expanded radius
            # BUT ONLY if the user didn't explicitly specify a radius
            original_radius = radius
            auto_expanded = False
            
            if len(combined_results) < min_results and radius < 50 and radius_data.get("Radius") == "10":
                auto_expanded = True
                # Try with progressively larger radii until we get enough results or hit the max
                for expanded_radius in [25, 50]:
                    if expanded_radius <= radius:
                        continue
                        
                    logger.info(f"Not enough results ({len(combined_results)}). Expanding search to {expanded_radius} miles.")
                    radius = expanded_radius
                    
                    # Search again with expanded radius
                    expanded_facilities = await self.search_facilities(lat, lon, activity, radius, api_limit)
                    expanded_rec_areas = await self.search_recreation_areas(lat, lon, activity, radius, api_limit)
                    
                    # Log the number of results found with expanded radius
                    logger.info(f"Found {len(expanded_facilities)} facilities and {len(expanded_rec_areas)} recreation areas with expanded radius {radius}")
                    
                    # Clear previous results and add new ones
                    combined_results = []
                    
                    # Add facilities with source
                    for facility in expanded_facilities:
                        facility['source'] = 'facility'
                        combined_results.append(facility)
                        
                    # Add rec areas with source
                    for rec_area in expanded_rec_areas:
                        rec_area['source'] = 'rec_area'
                        combined_results.append(rec_area)
                        
                    # If we have enough results now, break
                    if len(combined_results) >= min_results:
                        break
            
            # Sort combined results by distance if available
            if combined_results:
                # Create a function to safely extract distance
                def get_distance(item):
                    if 'RecAreaDistance' in item:
                        return float(item.get('RecAreaDistance', float('inf')))
                    elif 'FacilityDistance' in item:
                        return float(item.get('FacilityDistance', float('inf')))
                    else:
                        return float('inf')
                
                combined_results.sort(key=get_distance)
            
            # Take up to 10 results to ensure variety
            combined_results = combined_results[:10]
            
            # Log the final number of results
            logger.info(f"Final combined results: {len(combined_results)}")
            
            # Summarize the results
            if not combined_results:
                search_summary = f"Looking for {activity if activity != 'none' else 'recreational activities'} near {location} within {radius} miles"
                no_results = True
            else:
                # If we expanded the radius automatically, mention it in the summary ONLY if we actually expanded
                if auto_expanded and radius > original_radius:
                    search_summary = f"Looking for {activity if activity != 'none' else 'recreational activities'} near {location} within {original_radius} miles"
                else:
                    search_summary = f"Looking for {activity if activity != 'none' else 'recreational activities'} near {location} within {radius} miles"
                no_results = False
            
            # Add delay before final API call
            await asyncio.sleep(1.5)
            response_text = await self.summarize_results(search_summary, combined_results)
            
            # If no results were found, add a suggestion to try again with expanded search
            if no_results:
                response_text += "\n\nI couldn't find any results matching your criteria. You can try:\n"
                response_text += "- Reply with 'expand search' to try again with a wider search area\n"
                response_text += "- Reply with 'expand search to 30' to specify a custom search radius (up to 50 miles)\n"
                response_text += "- Try a different location or activity"
            
            # Truncate if longer than Discord's limit
            if len(response_text) > 1990:
                response_text = response_text[:1990] + "..."
            
            # Store in conversation history
            self.conversation_history[user_id].append({
                "query": user_query,
                "response": response_text,
                "no_results": no_results,
                "location": location,
                "activity": activity
            })
            
            # Limit history size
            if len(self.conversation_history[user_id]) > self.max_history_length:
                self.conversation_history[user_id].pop(0)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error in run method: {str(e)}")
            return "I encountered an error while processing your request. Please try again later."
