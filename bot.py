import os
import discord
import logging
import traceback
from discord.ext import commands
from dotenv import load_dotenv
from agent import MistralAgent

PREFIX = "!"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("discord_bot.log")
    ]
)
logger = logging.getLogger("discord")

# Load the environment variables
load_dotenv()

# Create the bot with all intents
# The message content and members intent must be enabled in the Discord Developer Portal for the bot to work.
intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# Import the Mistral agent from the agent.py file
agent = MistralAgent()

# Get the token from the environment variables
token = os.getenv("DISCORD_TOKEN")


@bot.event
async def on_ready():
    """
    Called when the client is done preparing the data received from Discord.
    Prints message on terminal when bot successfully connects to discord.

    https://discordpy.readthedocs.io/en/latest/api.html#discord.on_ready
    """
    logger.info(f"{bot.user} has connected to Discord!")
    await bot.change_presence(activity=discord.Game(name="Recreation Guide | !info"))


@bot.event
async def on_message(message: discord.Message):
    """
    Called when a message is sent in any channel the bot can see.

    https://discordpy.readthedocs.io/en/latest/api.html#discord.on_message
    """
    # Don't delete this line! It's necessary for the bot to process commands.
    await bot.process_commands(message)

    # Ignore messages from self or other bots to prevent infinite loops.
    if message.author.bot or message.content.startswith("!"):
        return

    # Show typing indicator while processing
    async with message.channel.typing():
        try:
            # Process the message with the agent
            logger.info(f"Processing message from {message.author}: {message.content}")
            
            # Send a temporary message to indicate processing for longer queries
            if len(message.content) > 50:
                temp_msg = await message.reply("I'm searching for recreation information. This might take a moment...")
            else:
                temp_msg = None
                
            # Get response from agent
            response = await agent.run(message)
            
            # Delete temporary message if it exists
            if temp_msg:
                await temp_msg.delete()
                
            # Check if the response is a list of chunks
            if isinstance(response, list):
                # Send each chunk as a separate message
                for chunk in response:
                    await message.channel.send(chunk)
            else:
                # Send the response as a single message
                await message.channel.send(response)
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            logger.error(traceback.format_exc())
            await message.reply("I encountered an error while processing your request. Please try again later.")


# Commands
@bot.command(name="info", help="Shows help information for the recreation bot.")
async def info_command(ctx):
    embed = discord.Embed(
        title="Recreation Bot Help",
        description="I can help you find recreational activities and facilities in the United States!",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name="How to use",
        value="Just ask me about recreational activities in a specific location. For example:\n"
              "- Where can I go hiking near Seattle?\n"
              "- Show me 3 campgrounds within 20 miles of Denver\n"
              "- What are the fishing spots in Miami?\n"
              "- Are there any hiking trails near Boston?",
        inline=False
    )
    
    embed.add_field(
        name="Location Memory",
        value="WanderBot can remember your location for easier follow-up questions:\n"
              "1. First, tell me your location: \"I'm in Portland, OR\" or \"My location is Austin, TX\"\n"
              "2. Then you can simply ask: \"Where can I go hiking near me?\"\n"
              "3. WanderBot will remember your location for future queries\n"
              "4. You can change your location anytime by mentioning a new one",
        inline=False
    )
    
    embed.add_field(
        name="Available Activities",
        value="BIKING, CLIMBING, CAMPING, FISHING, HIKING, HUNTING, WINTER SPORTS, WATER SPORTS, "
              "RECREATIONAL VEHICLES, WILDLIFE VIEWING, and more!",
        inline=False
    )
    
    embed.add_field(
        name="Commands",
        value=f"`{PREFIX}welcome` - Learn about WanderBot's mission to get you outdoors\n"
              f"`{PREFIX}info` - Shows this info message\n"
              f"`{PREFIX}ping` - Checks if the bot is responsive",
        inline=False
    )
    
    embed.set_footer(text="Data provided by Recreation.gov RIDB API")
    
    await ctx.send(embed=embed)


@bot.command(name="ping", help="Checks if the bot is responsive.")
async def ping(ctx):
    await ctx.send(f"Pong! Bot latency is {round(bot.latency * 1000)}ms")


@bot.command(name="welcome", help="Explains WanderBot's purpose and mission.")
async def welcome_command(ctx):
    embed = discord.Embed(
        title="🌲 Welcome to WanderBot! 🏞️",
        description="Your guide to breaking the cycle of endless scrolling and getting outdoors!",
        color=discord.Color.green()
    )
    
    embed.add_field(
        name="Our Mission",
        value="WanderBot helps break the cycle of endless scrolling and gets people off their screens and into the outdoors! "
              "We all know how easy it is to lose track of time online, but spending time outside is crucial for our well-being. "
              "This bot provides gentle encouragement for movement and outdoor exploration.",
        inline=False
    )
    
    embed.add_field(
        name="How WanderBot Helps",
        value="• Finds hiking trails, camping spots, and other outdoor activities near you\n"
              "• Suggests recreational facilities based on your interests and location\n"
              "• Provides information about amenities, features, and what to expect\n"
              "• Makes it easy to discover new outdoor adventures without endless research",
        inline=False
    )
    
    embed.add_field(
        name="Benefits of Outdoor Time",
        value="• Reduces stress and anxiety\n"
              "• Improves mood and mental health\n"
              "• Increases physical activity\n"
              "• Enhances creativity and focus\n"
              "• Creates meaningful experiences and memories",
        inline=False
    )
    
    embed.add_field(
        name="Get Started",
        value="Just ask me about recreational activities in your area! For example:\n"
              "• \"Where can I go hiking near Portland?\"\n"
              "• \"Find camping spots within 30 miles of Austin\"\n"
              "• \"What are some outdoor activities in Chicago?\"",
        inline=False
    )
    
    embed.set_footer(text="Type !info for more specific usage instructions")
    
    # Add a nature image to make the embed more appealing
    embed.set_thumbnail(url="https://i.imgur.com/8tRFLp8.png")  # Generic outdoor/hiking icon
    
    await ctx.send(embed=embed)


# Error handling for commands
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send(f"Command not found. Try `{PREFIX}info` for a list of commands.")
    else:
        logger.error(f"Command error: {str(error)}")
        await ctx.send(f"An error occurred: {str(error)}")


# Start the bot, connecting it to the gateway
if __name__ == "__main__":
    try:
        bot.run(token)
    except Exception as e:
        logger.critical(f"Failed to start bot: {str(e)}")
        logger.critical(traceback.format_exc())
