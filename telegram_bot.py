# from telegram import Update, Bot
# from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
# import pickle
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import openai
# from database import save_chat, get_chat_history
# from fastapi import FastAPI, Request
# import telegram
# import os
# from dotenv import load_dotenv
# from queue import Queue
# from contextlib import asynccontextmanager
# import uvicorn

# load_dotenv()

# # Load API Keys from Environment Variables
# TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# WEBHOOK_URL = os.getenv("WEBHOOK_URL")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# # PORT = int(os.getenv("PORT", 8000))
# try:
#     PORT = int(os.getenv("PORT", "8000"))
# except ValueError:
#     PORT = 8000

# # Validate required environment variables
# if not TOKEN or not OPENAI_API_KEY:
#     raise ValueError("❌ Missing TELEGRAM_BOT_TOKEN or OPENAI_API_KEY in environment variables.")

# # Initialize FastAPI app
# app = FastAPI()

# # Initialize Telegram Bot
# bot = Bot(token=TOKEN)
# application = Application.builder().token(TOKEN).build()
# print(application)

# # Load FAISS index & stored embeddings
# index = faiss.read_index("faiss_hnsw_index.bin")
# with open("embeddings.pkl", "rb") as f:
#     chunks, embeddings = pickle.load(f)

# # Load Sentence Transformer model
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# # ✅ Function to search similar verses from FAISS
# def search_gita(query, top_k=3):
#     query_embedding = model.encode([query], convert_to_numpy=True)
#     distances, indices = index.search(query_embedding, top_k)
#     return [chunks[i] for i in indices[0]]


# # ✅ Function to generate OpenAI answer
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

#     return response.choices[0].message.content


# # ✅ Telegram Command: /start
# async def start(update: Update, context: CallbackContext) -> None:
#     await update.message.reply_text("🙏 Welcome to Bhagavad Gita Q&A Bot! Ask any life-related question.")


# # ✅ Telegram Message Handler
# async def handle_message(update: Update, context: CallbackContext) -> None:
#     user_id = str(update.message.chat_id)
#     user_query = update.message.text
#     response = await generate_answer(user_query)
#     save_chat(user_id, user_query, response)
#     await update.message.reply_text(response)


# # ✅ Add Telegram Bot Handlers
# application.add_handler(CommandHandler("start", start))
# application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))


# # ✅ Webhook Route (No asyncio.run() now)
# @app.post("/webhook")
# async def webhook(request: Request):
#     print("Request: ", request.json())
#     update = telegram.Update.de_json(await request.json(), application.bot)
#     print(update)
#     await application.update_queue.put(update)
#     return {"status": "ok"}


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # ✅ Dynamically get Railway URL with HTTPS
#     webhook_url = f"https://{os.getenv('RAILWAY_PUBLIC_DOMAIN')}/webhook"
#     print(f"✅ Setting Webhook to: {webhook_url}")
#     print(f"Telegram token: {os.getenv('TELEGRAM_BOT_TOKEN')}")
#     # ✅ Set Telegram Webhook
#     await application.bot.setWebhook(webhook_url)
#     print(f"✅ Webhook set successfully to {webhook_url}")

#     yield

#     # ✅ Delete Webhook on shutdown
#     print("❌ Shutting down... Deleting Webhook")
#     await application.bot.deleteWebhook()


# # ✅ Register Lifespan (this replaces @app.on_event("startup"))
# app.router.lifespan_context = lifespan


# # ✅ Health Check Route
# @app.get("/")
# async def health_check():
#     return {"status": "Bot is running successfully!"}

# # ✅ Run Uvicorn Server (for Railway)
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=PORT)



from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from database import save_chat, get_chat_history
from fastapi import FastAPI, Request
import telegram
import os
from decouple import config
from queue import Queue
from contextlib import asynccontextmanager
import uvicorn
import asyncio

# Load API Keys from Environment Variables
TOKEN = config("TELEGRAM_BOT_TOKEN")
print(f"Token: {TOKEN}")
WEBHOOK_URL = config("WEBHOOK_URL")
OPENAI_API_KEY = config("OPENAI_API_KEY")

# Validate required environment variables
if not TOKEN or not OPENAI_API_KEY:
    raise ValueError("❌ Missing TELEGRAM_BOT_TOKEN or OPENAI_API_KEY in environment variables.")


# Load FAISS index & stored embeddings
index = faiss.read_index("faiss_hnsw_index.bin")
with open("embeddings.pkl", "rb") as f:
    chunks, embeddings = pickle.load(f)

# Load Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ✅ Function to search similar verses from FAISS
def search_gita(query, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]


# ✅ Function to generate OpenAI answer
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


# ✅ Telegram Command: /start
# async def start(update: Update, context: CallbackContext) -> None:
#     await update.message.reply_text("🙏 Welcome to Bhagavad Gita Q&A Bot! Ask any life-related question.")


# # ✅ Telegram Message Handler
# async def handle_message(update: Update, context: CallbackContext) -> None:
#     user_id = str(update.message.chat_id)
#     user_query = update.message.text
#     response = await generate_answer(user_query)
#     save_chat(user_id, user_query, response)
#     await update.message.reply_text(response)


# # ✅ Add Telegram Bot Handlers
# application.add_handler(CommandHandler("start", start))
# application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))


# # ✅ Webhook Route
# @app.post("/api/webhook")
# async def webhook(request: Request):
#     try:
#         update_data = await request.json()
#         update = telegram.Update.de_json(update_data, application.bot)
#         await application.process_update(update)  # Directly process update
#         return {"status": "ok"}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}



# # ✅ Health Check Route
# @app.get("/")
# async def health_check():
#     return {"status": "Bot is running successfully!"}

# # ✅ Main entry point for Vercel
# def handler(request):
#     return app

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# Handle the /start command
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("🙏 Welcome to Bhagavad Gita Q&A Bot!\n\nAsk any life-related question, and I'll respond based on the Bhagavad Gita.")


# Handle incoming messages (Q&A)
async def handle_message(update: Update, context: CallbackContext) -> None:
    user_query = update.message.text
    response = await generate_answer(user_query)
    await update.message.reply_text(response)


# Main function to run the bot using polling
def main():
    # Initialize the Telegram Bot
    app = Application.builder().token(TOKEN).build()

    # Add Command and Message Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot with polling
    print("🤖 Telegram Bot is running using Polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
