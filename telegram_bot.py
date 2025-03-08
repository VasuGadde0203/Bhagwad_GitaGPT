# from telegram import Update
# from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
# import pickle
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import openai
# import asyncio
# from database import save_chat, get_chat_history  # Import DB functions
# from flask import Flask, request
# import telegram
# import os
# from dotenv import load_dotenv

# load_dotenv()

# print("🔍 Checking environment variables in Railway...")
# print(os.environ)  # Print all environment variables


# # Load API Keys from Environment Variables
# TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# WEBHOOK_URL = os.getenv("WEBHOOK_URL")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# print(f"Token: {TOKEN[:3]}")
# print(f"Open AI api key: {OPENAI_API_KEY[:3]}")

# # Validate required environment variables
# if not TOKEN or not OPENAI_API_KEY:
#     raise ValueError("❌ Missing TELEGRAM_BOT_TOKEN or OPENAI_API_KEY in environment variables.")

# # Initialize Flask app
# app = Flask(__name__)
# bot = telegram.Bot(token=TOKEN)

# # Load FAISS index & stored embeddings
# index = faiss.read_index("faiss_hnsw_index.bin")
# with open("embeddings.pkl", "rb") as f:
#     chunks, embeddings = pickle.load(f)

# # Load Sentence Transformer model
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# def search_gita(query, top_k=3):
#     """Search Bhagavad Gita for relevant verses based on a query."""
    
#     # Convert query to an embedding
#     query_embedding = model.encode([query], convert_to_numpy=True)

#     # Search FAISS index
#     distances, indices = index.search(query_embedding, top_k)

#     # Retrieve top matching text chunks
#     return [chunks[i] for i in indices[0]]

# async def generate_answer(query):
#     """Generate a response using GPT-4 based on retrieved verses."""
    
#     # Retrieve relevant verses
#     matching_verses = search_gita(query)
    
#     # Combine verses into context for GPT-4
#     context = "\n".join(matching_verses)

#     # Create a prompt for GPT-3.5 turbo
#     prompt = f"""
#     You are a Bhagavad Gita expert. Answer the following question based on these verses:
    
#     Context:
#     {context}
    
#     Question: {query}
    
#     Answer in a simple, clear, and helpful way and keep it short under 120 words.
#     """
    
#     # Call OpenAI GPT-4
#     response = openai.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "system", "content": "You are a Bhagavad Gita scholar."},
#                   {"role": "user", "content": prompt}],
#         max_tokens=200
#     )

#     return await asyncio.to_thread(str, response.choices[0].message.content)

# @app.route("/webhook", methods=["POST"])
# def webhook():
#     """Handle incoming Telegram messages via Webhook."""
#     update = telegram.Update.de_json(request.get_json(), bot)
#     asyncio.run(app.telegram_application.process_update(update))
#     return "OK", 200

# async def start(update: Update, context: CallbackContext) -> None:
#     await update.message.reply_text("🙏 Welcome to Bhagavad Gita Q&A Bot! Ask any life-related question.")

# async def handle_message(update: Update, context: CallbackContext) -> None:
#     user_id = str(update.message.chat_id)
#     user_query = update.message.text

#     # Generate response from Bhagavad Gita bot
#     response = await generate_answer(user_query)

#     # Save chat history in MongoDB
#     save_chat(user_id, user_query, response)

#     # Send response to user
#     await update.message.reply_text(response)

# def main():
#     """Start the bot with webhook handling on Railway."""
#     application = Application.builder().token(TOKEN).build()
#     application.add_handler(CommandHandler("start", start))
#     application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

#     # Assign application globally
#     app.telegram_application = application

#     # Start Flask app (Railway will auto-assign a port)
#     app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))

# if __name__ == "__main__":
#     main()



from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from database import save_chat, get_chat_history
from flask import Flask, request
import telegram
import os
from dotenv import load_dotenv
from queue import Queue
import asyncio

load_dotenv()

# Load API Keys from Environment Variables
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Validate required environment variables
if not TOKEN or not OPENAI_API_KEY:
    raise ValueError("❌ Missing TELEGRAM_BOT_TOKEN or OPENAI_API_KEY in environment variables.")

# Initialize Flask app
app = Flask(__name__)
# bot = telegram.Bot(token=TOKEN)
application = Application.builder().token(TOKEN).update_queue(Queue()).build()

# Load FAISS index & stored embeddings
index = faiss.read_index("faiss_hnsw_index.bin")
with open("embeddings.pkl", "rb") as f:
    chunks, embeddings = pickle.load(f)

# Load Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def search_gita(query, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]


async def generate_answer(query):
    matching_verses = search_gita(query)
    context = "\n".join(matching_verses)
    prompt = f"""
    You are a Bhagavad Gita expert. Answer the following question based on these verses:

    Context:
    {context}

    Question: {query}

    Answer in a clear and concise manner under 120 words.
    """

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a Bhagavad Gita scholar."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )

    return await asyncio.to_thread(str, response.choices[0].message.content)


@app.route("/webhook", methods=["POST"])
def webhook():
    update = telegram.Update.de_json(request.get_json(force=True), application.bot)
    # app.telegram_application.update_queue.put(update)
    asyncio.run(application.update_queue.put(update))
    return "OK", 200


async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("🙏 Welcome to Bhagavad Gita Q&A Bot! Ask any life-related question.")


async def handle_message(update: Update, context: CallbackContext) -> None:
    user_id = str(update.message.chat_id)
    user_query = update.message.text
    response = await generate_answer(user_query)
    save_chat(user_id, user_query, response)
    await update.message.reply_text(response)


async def main():
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Bind the application to Flask
    app.telegram_application = application
    await application.bot.setWebhook(f"{WEBHOOK_URL}/webhook")

    # Run Flask
    # app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
    
    # ✅ Run Flask app in a separate asyncio task
    loop = asyncio.get_running_loop()
    task = loop.run_in_executor(None, app.run, "0.0.0.0", int(os.getenv("PORT", 5000)))
    await task


if __name__ == "__main__":
    asyncio.run(main())
