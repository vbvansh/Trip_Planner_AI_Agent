import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Fetch API Key
api_key = os.getenv("GROQ_API_KEY")

# Check if API key is loaded correctly
if not api_key:
    raise ValueError("❌ API Key not found! Check your .env file.")

# Initialize Groq client
client = Groq(api_key=api_key)

# Test Model
completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # Use correct model name
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    temperature=1,
    max_tokens=512,
    top_p=1,
    stream=False
)

print(completion.choices[0].message.content)
