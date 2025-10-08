from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm=ChatGroq(groq_api_key=GROQ_API_KEY,model_name="gemma2-9b-it",temperature=0.1,max_tokens=1024)

# Conversation memory: store chat history in-memory and pass it to the RAG chain
memory = ConversationBufferMemory(return_messages=True)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        # Insert the chat history between system and user messages so the LLM
        # can use previous turns when answering.
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    # Pass the current chat history from memory into the retrieval chain.
    # The chain expects `chat_history` to be a list of messages (BaseMessage),
    # which ConversationBufferMemory.buffer_as_messages provides when
    # return_messages=True.
    response = rag_chain.invoke({"input": msg, "chat_history": memory.buffer_as_messages})
    print("Response : ", response["answer"])

    # Save this turn to memory so it will be available for future turns.
    try:
        memory.save_context({"input": msg}, {"answer": response["answer"]})
    except Exception:
        # Memory saving shouldn't break the request; ignore failures silently.
        pass
    return str(response["answer"])


@app.route("/clear", methods=["POST", "GET"])
def clear_memory():
    """Clear the in-memory conversation history (useful during development)."""
    try:
        memory.clear()
        return jsonify({"status": "ok", "message": "memory cleared"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)