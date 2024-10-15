import os
import json
import subprocess
from typing import List
from langchain_chroma import Chroma  # Updated import for Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import uvicorn
from update_database import get_embedding_function
load_dotenv()

CONNECT_GROQ = False

if CONNECT_GROQ:
    # # Get the API key from the environment variable
    groq_api_key_ = os.getenv("GROQ_API_KEY")
    print(f'groq_api_key_: {groq_api_key_}')
    if groq_api_key_ is None:
        print("GROQ API key not found. Please check your .env file.")
        print("Connecting to Ollama API...")
        llm = Ollama(model="llama3.1:8b")
    else:
        print("Connecting to Groq API...")
        ## for running llm model from Groq
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            temperature=0, 
            groq_api_key=groq_api_key_, 
            model_name="llama-3.1-70b-versatile"
        )
else:
    ## for running llm model from Ollama
    print("Connecting to Ollama API...")
    llm = Ollama(model="llama3.1:8b")
        


app = FastAPI()

CHROMA_PATH = "database"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Manage connected WebSocket clients and their conversation histories
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.conversation_context = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Initialize conversation context for the client
        self.conversation_context[client_id] = []

    def disconnect(self, websocket: WebSocket, client_id: str):
        self.active_connections.remove(websocket)
        # Remove the conversation context for the client
        if client_id in self.conversation_context:
            del self.conversation_context[client_id]

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    def update_context(self, client_id: str, query_text: str, response_data):
        if client_id in self.conversation_context:
            # Append the query and response to the conversation history
            if isinstance(response_data, dict):
                self.conversation_context[client_id].append({"query": query_text, "response": response_data["response_text"]})
            elif isinstance(response_data, str):    
                self.conversation_context[client_id].append({"query": query_text, "response": response_data})

manager = ConnectionManager()

# Query function for RAG (Retrieval-Augmented Generation)
def query_rag(query_text: str, client_id: str):
    # Retrieve and store context for the conversation
    previous_context = "\n\n---\n\n".join([item['response'] for item in manager.conversation_context.get(client_id, [])])

    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    full_context = previous_context + "\n\n" + context_text if previous_context else context_text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=full_context, question=query_text)

    response_text = llm.invoke(prompt)
    
    sources_list = [doc.metadata.get("id", None) for doc, _score in results]
    sources = '\n '.join(sources_list)

    formatted_response = {
        "prompt": prompt,
        "sources": sources
    }

    if isinstance(response_text, str):
        ## Create structured response -- if using Ollama:
        formatted_response["response_text"] = response_text
        print(f'str, {response_text}')
    else:
        ## Create structured response -- if using ChatGroq:
        formatted_response["response_text"] = response_text.content
        print(f'dic, {response_text.content}')

    return formatted_response

# WebSocket route to manage real-time chat
@app.websocket("/ws/chat/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            response_data = query_rag(data, client_id)  # Get structured response
            manager.update_context(client_id, data, response_data)
            
            # Send the response as a JSON string
            await websocket.send_text(json.dumps(response_data))
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)

# Serve your HTML file and static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_html():
    with open("static/chat_box.html") as f:  # Make sure the path matches your HTML file name
        return f.read()



# Existing FastAPI app and imports...

@app.post("/update-database")
async def update_database():
    try:
        # Run the update_database.py script
        subprocess.run(["python", "update_database.py"], check=True)
        
        return {"message": "Database updated successfully! Please restart the server to apply changes."}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail="Database update failed.")


# Run the application
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=80)


