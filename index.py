import json
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

import gradio as gr

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", streaming=True)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

#%% 
# Load horoscope data
def load_horoscope_data(file_path="data/horoscopes.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["signs"]

# %%
# Create RAG
def build_rag():
    signs = load_horoscope_data()
    
    documents = []
    for sign in signs:
        content = f"""Sign: {sign['sign']}
            Dates: {sign['dates']}
            Element: {sign['element']}
            Ruling Planet: {sign['planet']}
            Personality: {sign['personality']}"""
        
        documents.append({
            "page_content": content,
            "metadata": {"sign": sign['sign']}
        })

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = text_splitter.create_documents(
        [doc["page_content"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents]
    )

    vectorstore = Chroma.from_documents(splits, embeddings, collection_name="horoscope_rag")
    return vectorstore.as_retriever(search_kwargs={"k": 6})

#%% 
# These contexts define the personalitiy of how the AI will speak
system_prompt = """you are a astrologer, a gentle, mysterious, and poetic one.

You are bilingual and naturally mix English with Thai, just like many people in Thailand do.
- Main language is English
- You naturally weave in Thai words and short phrases (using real Thai script)

Important Instructions:
- Connect new answers with previous messages naturally.
- Stay completely in character as astrologer. Never break role.
- Answer using only the provided context.
- Always mention the zodiac sign clearly when relevant."""

# %%
# Build Chain
retriever = build_rag()

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt + "\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# %%
# STREAMING FUNCTION
def stream_response(message: str, history):
    if not history or len(history) == 0:
        welcome = """Hello dear soul... The stars have brought you here tonight. May I know your zodiac sign?
        **Here are all the zodiac signs:**
        **Aries** (March 21 - April 19)  
        **Taurus** (April 20 - May 20)  
        **Gemini** (May 21 - June 20)  
        **Cancer** (June 21 - July 22)  
        **Leo** (July 23 - August 22)  
        **Virgo** (August 23 - September 22)  
        **Libra** (September 23 - October 22)  
        **Scorpio** (October 23 - November 21)  
        **Sagittarius** (November 22 - December 21)  
        **Capricorn** (December 22 - January 19)  
        **Aquarius** (January 20 - February 18)  
        **Pisces** (February 19 - March 20)"""
        yield welcome
        return

    chat_history = []
    # Fixed unpacking for latest Gradio (handles 2 or 4 values)
    for turn in history:
        if isinstance(turn, (list, tuple)):
            # New Gradio format can have 4 items: (user, assistant, user_file, assistant_file)
            user_msg = turn[0] if len(turn) > 0 else None
            assistant_msg = turn[1] if len(turn) > 1 else None
            
            if user_msg:
                chat_history.append(HumanMessage(content=str(user_msg)))
            if assistant_msg:
                chat_history.append(AIMessage(content=str(assistant_msg)))

    # Add current user message
    if message:
        chat_history.append(HumanMessage(content=message))

    # Stream the response
    partial_message = ""
    for chunk in rag_chain.stream({
        "input": message, 
        "chat_history": chat_history
    }):
        if "answer" in chunk:
            partial_message += chunk["answer"]
            yield partial_message

# %%
# i
demo = gr.ChatInterface(
        title="AI HOROSCOPE",
        description="A Star Whisper, Powered by Gemini 2.5 Flash (Free)",

    fn=stream_response,
    textbox=gr.Textbox(
        placeholder="พูดคุยกับ Whisper... (Speak to Whisper...)",
        container=False,
        autoscroll=True,
        scale=5
    ),
    examples=[
        ["Hello"],
    ],
    fill_height=True,

)

demo.launch(
    debug=True, 
    share=True,
    server_name="0.0.0.0",   # Optional: makes it accessible on local network
    show_error=True
)
