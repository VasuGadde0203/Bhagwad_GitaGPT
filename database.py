from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import os 

# MongoDB Connection (Replace with your credentials)
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["GitaBot"]  # Database name
collection = db["chat_history"]  # Collection name

def save_chat(user_id, user_message, bot_response):
    """Save chat history to MongoDB."""
    chat_entry = {
        "user_id": user_id,
        "user_message": user_message,
        "bot_response": bot_response,
        "timestamp": datetime.utcnow()
    }
    collection.insert_one(chat_entry)

def get_chat_history(user_id, limit=5):
    """Retrieve last 5 messages of a user."""
    chats = collection.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
    return [(chat["user_message"], chat["bot_response"]) for chat in chats]
