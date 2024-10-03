# DatabaseDialogue
This repository hosts a local chat application that connects to a vectorized Chroma database, generated from user-provided documents. The application enables real-time updates and delivers dynamic responses using local AI models, such as Llama. Users can send messages through a FastAPI interface, and the application processes these messages using the latest data from the database. The application allows for updating the database and requires a server restart to reflect the changes in the chat context. 

### Key Features:
- Real-time Chat: Communicate with the system using your documents as the knowledge base.
- Dynamic Query Processing: The system provides responses based on the latest database state.
- Database Update Functionality: Users can update the Chroma database in real-time.
- Responsive Frontend: Simple, user-friendly HTML/CSS chat interface.
- Custom AI Models: Easily switch between local AI models like Llama or connect to online models using APIs.

## Requirements

- Python 3.7 or higher
- FastAPI
- Uvicorn
- WebSocket support
- A local database (SQLite, PostgreSQL, etc.) or a JSON file

## Set-up

### 1. Clone the repository:
```bash
git clone https://github.com/yourusername/local-chat-with-database.git
cd DatabaseDialogue
```

### 2. Install the dependencies using:
```bash
pip install -r requirements.txt
```

### 3. Run Ollama or Connect to your llm model:    
to run with Ollama:
a. download and install ollama from (https://ollama.com/download)
b. download and install llama 3.1 (or any other prefered model) and llama text embedding by: 
```bash
Ollama pull llama3.1:8b
Ollama pull nomic-embed-text
```

To use online llm models using APIs we first need to get an API_KEY from here: https://console.groq.com/keys. 
Inside app/.env update the value of GROQ_API_KEY with the API_KEY you created. 

### 4. Run the Application:    

To run the FastAPI application locally: 
```bash
uvicorn app:app --reload
```

This will start the FastAPI app on http://127.0.0.1:8000. You can now open your browser and access the chat interface.

### 5. Access the Chat Interface:    

Open your browser and navigate to: http://localhost:8000/

When the user asks a question, the response will appear in three parts:

1. Response to User's Question: The main response generated by the AI.
2. Source Information: Presented as Page Source : Page Number : Chunk 3.Index to indicate where the information was retrieved from.
3. Source Text: The actual text from the documents that was used to generate the response.

## Updating the Database

### 1. Update Database: 
You can update the Chroma database by clicking the "Update Database" button in the chat interface. This will run the update_database.py script.


### 2 Restart the Server: 
After updating the database, you need to restart the FastAPI server to reflect the changes. You can use the terminal to stop the server (Ctrl+C) and then restart it using:then restart it using: 
```bash
uvicorn app:app --reload
```

## Customization
<!-- ### Models:  -->
You can modify the app.py file to integrate different AI models for responses (e.g., replace Llama with another model).

### Database: 
The database is vectorized using Chroma and can be updated by modifying the data sources used in the update_database.py script.
The Chroma database creation is inspierd by https://github.com/pixegami