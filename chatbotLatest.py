# ========== IMPORTS ==========
import os
import glob
import json
from dotenv import load_dotenv
import gradio as gr
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go

# ========== CONFIGURATION ==========
load_dotenv(override=True)

# Use FREE embeddings instead of OpenAI
MODEL = "gpt-3.5-turbo"  # Changed to reliable model
db_name = "vector_db"

# ========== PHASE 1: BUILD KNOWLEDGE BASE ==========
print("🔄 Step 1: Loading and processing documents...")

# Read in documents using LangChain's loaders
folders = glob.glob("knowledge-base/*")
text_loader_kwargs = {'encoding': 'utf-8'}

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

print(f"✅ Loaded {len(documents)} documents")

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
print(f"📄 Document types found: {', '.join(doc_types)}")
print(f"✅ Total chunks created: {len(chunks)}")

# ========== SETUP EMBEDDINGS ==========
print("🔄 Setting up embeddings...")

try:
    # HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Test embedding
    test_embedding = embeddings.embed_query("Test message")
    print(f"✅ HuggingFace Embeddings working! Dimension: {len(test_embedding)}")

except Exception as e:
    print(f"❌ Error with HuggingFace embeddings: {e}")
    print("🔄 Trying OpenAI embeddings as fallback...")
    
    try:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        print("✅ Using OpenAI embeddings as fallback")
    except Exception as e2:
        print(f"❌ No embedding method available: {e2}")
        raise Exception("No working embeddings found")

# ========== CREATE VECTOR DATABASE ==========
print("🔄 Creating vector database...")

vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory=db_name
)

print(f"✅ Vectorstore created with {vectorstore._collection.count()} documents")

# ========== VISUALIZATION (Optional) ==========
print("🔄 Creating visualizations...")

try:
    collection = vectorstore._collection
    result = collection.get(include=['embeddings', 'documents', 'metadatas'])
    vectors = np.array(result['embeddings'])
    documents_text = result['documents']
    doc_types = [metadata['doc_type'] for metadata in result['metadatas']]
    
    # Create color mapping
    unique_doc_types = list(set(doc_types))
    color_map = {doc_type: f'rgb({hash(doc_type) % 255}, {hash(doc_type + "1") % 255}, {hash(doc_type + "2") % 255})' 
                 for doc_type in unique_doc_types}
    colors = [color_map[t] for t in doc_types]

    # 2D Visualization
    tsne_2d = TSNE(n_components=2, random_state=42)
    reduced_vectors_2d = tsne_2d.fit_transform(vectors)

    fig_2d = go.Figure(data=[go.Scatter(
        x=reduced_vectors_2d[:, 0],
        y=reduced_vectors_2d[:, 1],
        mode='markers',
        marker=dict(size=5, color=colors, opacity=0.8),
        text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents_text)],
        hoverinfo='text'
    )])

    fig_2d.update_layout(
        title='2D Document Embeddings Visualization',
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        width=800,
        height=600
    )
    fig_2d.show()

    # 3D Visualization
    tsne_3d = TSNE(n_components=3, random_state=42)
    reduced_vectors_3d = tsne_3d.fit_transform(vectors)

    fig_3d = go.Figure(data=[go.Scatter3d(
        x=reduced_vectors_3d[:, 0],
        y=reduced_vectors_3d[:, 1],
        z=reduced_vectors_3d[:, 2],
        mode='markers',
        marker=dict(size=5, color=colors, opacity=0.8),
        text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents_text)],
        hoverinfo='text'
    )])

    fig_3d.update_layout(
        title='3D Document Embeddings Visualization',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        width=900,
        height=700
    )
    fig_3d.show()
    
    print("✅ Visualizations created successfully!")

except Exception as e:
    print(f"⚠️ Visualization failed: {e}")
    print("Continuing with chatbot setup...")

# ========== CHAT SYSTEM SETUP ==========
print("🔄 Setting up chat system...")

# Initialize OpenAI client
openai_client = OpenAI()

# System prompt
system_message = (
    "You are a helpful and professional assistant for Intensity, a company providing "
    "cybersecurity, cloud, and AI solutions. "
    "Answer customer questions clearly and accurately. "
    "Use the provided context from company documents to answer questions. "
    "Keep answers concise, but you may provide more details if the user seems to need them. "
    "Maintain a courteous and friendly tone. "
    "If you don't know the answer based on the context, politely admit it and suggest contacting an Intensity expert for further help."
)

# Debug API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if openai_api_key:
    print(f"✅ OpenAI API Key exists and begins {openai_api_key[:8]}...")
else:
    print("❌ OpenAI API Key not set")

if anthropic_api_key:
    print(f"✅ Anthropic API Key exists and begins {anthropic_api_key[:7]}...")
else:
    print("❌ Anthropic API Key not set")

if google_api_key:
    print(f"✅ Google API Key exists and begins {google_api_key[:8]}...")
else:
    print("❌ Google API Key not set")

# ========== TOOL DEFINITIONS ==========
# Example pricing dictionary for Intensity's services (in INR)
service_prices = {
    "cloud storage basic": "₹8,000/month",
    "cloud storage enterprise": "₹41,000/month",
    "cloud migration": "₹82,000/project",
    "cybersecurity audit": "₹66,000",
    "cybersecurity monitoring": "₹25,000/month",
    "penetration testing": "₹1,25,000",
    "ai chatbot integration": "₹58,000/project",
    "ai analytics": "₹33,000/month",
    "ai custom solution": "Starting at ₹2,50,000"
}

def get_service_price(service_name: str) -> str:
    """
    Lookup tool for Intensity's services pricing (in INR).
    Takes a service name, normalizes it, and returns the price if available.
    """
    print(f"🔧 Tool get_service_price called for: {service_name}")
    service = service_name.lower().strip()
    return service_prices.get(
        service,
        "Price not available – please contact an Intensity expert for details."
    )

# Define tool schema
get_service_price_tool = {
    "type": "function",
    "function": {
        "name": "get_service_price",
        "description": "Look up the pricing for Intensity's cloud, cybersecurity, and AI services (in INR).",
        "parameters": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "The name of the Intensity service. Example: 'cloud storage basic', 'cybersecurity audit', 'ai chatbot integration'."
                }
            },
            "required": ["service_name"]
        }
    }
}

tools = [get_service_price_tool]

# ========== MAIN CHAT FUNCTION ==========
def chat_with_rag(message, history):
    """
    Enhanced chat function that uses RAG + tools
    message: str (new user message)
    history: list of messages in Gradio Chatbot format
    """
    print(f"💬 User message: {message}")
    
    # STEP 1: Search vector database for relevant information (RAG)
    try:
        relevant_docs = vectorstore.similarity_search(message, k=3)
        context = "\n\n".join([f"From {doc.metadata.get('doc_type', 'unknown')}:\n{doc.page_content}" 
                              for doc in relevant_docs])
        print(f"📚 Found {len(relevant_docs)} relevant documents")
    except Exception as e:
        print(f"❌ RAG search failed: {e}")
        context = "No relevant documents found."
    
    # STEP 2: Build messages with context
    messages = [
        {
            "role": "system", 
            "content": f"{system_message}\n\nIMPORTANT: Use this context from company documents when relevant:\n{context}"
        }
    ]
    
    # Add conversation history
    for entry in history:
        if isinstance(entry, tuple) and len(entry) == 2:
            user_msg, assistant_msg = entry
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    try:
        # STEP 3: Get response with tool support
        completion = openai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = completion.choices[0].message

        # STEP 4: Handle tool calls if needed
        if response_message.tool_calls:
            tool_call = response_message.tool_calls[0]
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name == "get_service_price":
                price_result = get_service_price(arguments['service_name'])
                response = f"The price for **{arguments['service_name']}** is {price_result}."
                print(f"💰 Price lookup result: {response}")
            else:
                response = "I don't know how to use that tool yet."
        else:
            # Normal assistant response
            response = response_message.content

        # Append to history and return
        history.append((message, response))
        print(f"🤖 Assistant response: {response[:100]}...")
        return history, ""  # Clear the input box
        
    except Exception as e:
        error_response = f"Sorry, I encountered an error: {str(e)}"
        print(f"❌ Chat error: {e}")
        history.append((message, error_response))
        return history, ""

# ========== GRADIO UI ==========
print("🚀 Building chat interface...")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Intensity AI Assistant
    **Your intelligent companion for company information and services**
    
    I can help you with:
    - 📄 Answering questions from company documents
    - 💰 Providing service pricing information
    - 🔍 Finding relevant information across all departments
    """)
    
    chatbot = gr.Chatbot(
        height=500,
        placeholder="Ask me anything about Intensity's services or company information...",
        show_copy_button=True
    )
    
    msg = gr.Textbox(
        placeholder="Type your question here...",
        label="Your Message",
        scale=7
    )
    
    with gr.Row():
        submit = gr.Button("🚀 Send", variant="primary", scale=1)
        clear = gr.Button("🧹 Clear Chat", scale=1)

    # Handle message submission
    submit.click(
        fn=chat_with_rag,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )
    
    # Allow pressing Enter to send message
    msg.submit(
        fn=chat_with_rag,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )
    
    # Clear chat history
    clear.click(lambda: None, None, chatbot, queue=False)
    
    gr.Markdown("""
    ---
    *Powered by LangChain • Chroma • HuggingFace • OpenAI*
    """)

# ========== APPLICATION LAUNCH ==========
if __name__ == "__main__":
    print("🎉 Starting Intensity AI Assistant...")
    print("🌐 Opening web interface...")
    demo.launch(
        share=False, 
        debug=True,
        show_error=True
    )
