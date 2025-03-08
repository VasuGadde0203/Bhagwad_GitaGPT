# from telegram import Update
# from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
# import pickle
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import openai
# from database import save_chat, get_chat_history
# from flask import Flask, request
# import telegram
# import os
# from dotenv import load_dotenv
# from queue import Queue
# import asyncio

# load_dotenv()

# # Load API Keys from Environment Variables
# TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# WEBHOOK_URL = os.getenv("WEBHOOK_URL")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# # Validate required environment variables
# if not TOKEN or not OPENAI_API_KEY:
#     raise ValueError("❌ Missing TELEGRAM_BOT_TOKEN or OPENAI_API_KEY in environment variables.")

# # Initialize Flask app
# app = Flask(__name__)
# # bot = telegram.Bot(token=TOKEN)
# application = Application.builder().token(TOKEN).update_queue(Queue()).build()

# # Load FAISS index & stored embeddings
# index = faiss.read_index("faiss_hnsw_index.bin")
# with open("embeddings.pkl", "rb") as f:
#     chunks, embeddings = pickle.load(f)

# # Load Sentence Transformer model
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# def search_gita(query, top_k=3):
#     query_embedding = model.encode([query], convert_to_numpy=True)
#     distances, indices = index.search(query_embedding, top_k)
#     return [chunks[i] for i in indices[0]]


# async def generate_answer(query):
#     matching_verses = search_gita(query)
#     context = "\n".join(matching_verses)
#     prompt = f"""
#     You are a Bhagavad Gita expert. Answer the following question based on these verses:

#     Context:
#     {context}

#     Question: {query}

#     Answer in a clear and concise manner under 120 words.
#     """

#     response = openai.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a Bhagavad Gita scholar."},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=200
#     )

#     return await asyncio.to_thread(str, response.choices[0].message.content)


# @app.route("/webhook", methods=["POST"])
# def webhook():
#     update = telegram.Update.de_json(request.get_json(force=True), application.bot)
#     # app.telegram_application.update_queue.put(update)
#     asyncio.run(application.update_queue.put(update))
#     return "OK", 200


# async def start(update: Update, context: CallbackContext) -> None:
#     await update.message.reply_text("🙏 Welcome to Bhagavad Gita Q&A Bot! Ask any life-related question.")


# async def handle_message(update: Update, context: CallbackContext) -> None:
#     user_id = str(update.message.chat_id)
#     user_query = update.message.text
#     response = await generate_answer(user_query)
#     save_chat(user_id, user_query, response)
#     await update.message.reply_text(response)


# async def main():
#     application.add_handler(CommandHandler("start", start))
#     application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

#     # Bind the application to Flask
#     app.telegram_application = application
#     await application.bot.setWebhook(f"{WEBHOOK_URL}/webhook")

#     # Run Flask
#     # app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
    
#     # ✅ Run Flask app in a separate asyncio task
#     loop = asyncio.get_running_loop()
#     task = loop.run_in_executor(None, app.run, "0.0.0.0", int(os.getenv("PORT", 5000)))
#     await task


# if __name__ == "__main__":
#     asyncio.run(main())



from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from database import save_chat, get_chat_history
from fastapi import FastAPI
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

# Initialize FastAPI app
app = FastAPI()
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


@app.post("/webhook")
async def webhook(update: dict):
    telegram_update = telegram.Update.de_json(update, application.bot)
    await application.update_queue.put(telegram_update)
    return {"status": "ok"}


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

    await application.bot.setWebhook(f"{WEBHOOK_URL}/webhook")

# if __name__ == "__main__":
asyncio.run(main())
