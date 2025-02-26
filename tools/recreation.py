# add recreation shite here

import httpx
import logging
import json

USER_AGENT = "weather-app/1.0"
WEATHER_API_BASE = "https://api.open-meteo.com/v1/forecast?current=temperature_2m,precipitation,weather_code&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch&timezone=America%2FLos_Angeles"


logger = logging.getLogger("discord")

MISTRAL_MODEL = "mistral-large-latest"

EXTRACT_LOCATION_PROMPT = """
Is this message explicitly requesting recreation information for a specific city/location?
If not, return {"location": "none"}.

Otherwise, return the full name of the city in JSON format.

Example:
Message: Where can I fish in Salt Lake City?
Response: {“location”: “Salt Lake City, UT”}

Message: What are the closest campgrounds to Bozeman?
Response: {“location”: “Bozeman, MT”}

Message: Are there hiking trails near Boston?
Response: {“location”: “Boston, MA”}

Message: Give me the hiking trails in Boulder.
Response: {“location”: “Boulder, CO”}

Message: I love hiking in sf!
Response: {“location”: “none”}

Message: Is camping fun in NYC?
Response: {“location”: “none”}
"""

EXTRACT_ACTIVITY_PROMPT = """
Is this message explicitly requesting recreation information for a specific activity?
If not, return {“ActivityName”: "none"}.

Otherwise, return the full name of the city in JSON format.

Example:
Message: Where can I fish in Salt Lake City?
Response: {“ActivityName”: “FISHING”}

Message: What are the closest campgrounds to Bozeman?
Response: {“ActivityName”: “CAMPING”}

Message: Are there hiking trails near Boston?
Response: {“ActivityName”: “HIKING”}

Message: Give me the hiking trails in Boulder.
Response: {“ActivityName”: “HIKING”}

Message: I love hiking in sf!
Response: {“ActivityName”: “none”}

Message: Is camping fun in NYC?
Response: {“ActivityName”: “none”}
"""

EXTRACT_RADIUS_PROMPT = """
Is this message explicitly requesting recreation information for a specific radius?
If not, return {“Radius”: “10”}.

Otherwise, return the radius specified. If the value is greater than 50, return 50.

Example:
Message: Where can I fish within 30 miles of  Salt Lake City?
Response: {“Radius”: “30”}

Message: What are the closest campgrounds to Bozeman?
Response: {“Radius”: “10”}

Message: Are there hiking trails within 60 miles of Boston?
Response: {“Radius”: “60”}

Message: Give me the hiking trails in Boulder.
Response: {“Radius”: “None”}

Message: I love hiking in sf!
Response: {“Radius”: “none”}

Message: Is camping fun in NYC?
Response: {“Radius”: “none”}
"""

EXTRACT_LIMIT_PROMPT = """
Is this message explicitly requesting a limit on the number of results to return?
If not, return {“Limit”: “5”}.

Otherwise, return the limit specified, with a max of ten. 

Example:
Message: Show me the 3 best places to fish near Salt Lake City.
Response: {“Limit”: “3”}

Message: What are the closest campgrounds to Bozeman?
Response: {“Limit”: “5”}

Message: Show me 11 hiking trails within 60 miles of Boston?
Response: {“Limit”: “10”}

Message: Give me the hiking trails in Boulder.
Response: {“Limit”: “5”}
"""


def _make_request(url: str):
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}

    try:
        response = httpx.Client().get(url, headers=headers, timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def seven_day_forecast(latitude: str, longitude: str):
    """Get the seven day forecast for a given location with latitude and longitude."""
    logger.info(f"Getting seven day forecast for {latitude}, {longitude}")
    url = f"{WEATHER_API_BASE}&latitude={latitude}&longitude={longitude}"
    data = _make_request(url)

    if data is None:
        return "Error fetching weather data"

    res_json = {
        "current": data["current"],
        "daily": {},
    }

    for i, time in enumerate(data["daily"]["time"]):
        max_temp = data["daily"]["temperature_2m_max"][i]
        min_temp = data["daily"]["temperature_2m_min"][i]
        precipitation = data["daily"]["precipitation_probability_max"][i]
        res_json["daily"][time] = {
            "weather_code": data["daily"]["weather_code"][i],
            "temperature_max": f"{max_temp}°F",
            "temperature_min": f"{min_temp}°F",
            "precipitation": f"{precipitation}%",
        }

    return json.dumps(res_json)