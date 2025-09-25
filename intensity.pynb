# imports
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

# Load environment variables in a file called .env
load_dotenv(override=True)

# Initialize OpenAI client
openai = OpenAI()
MODEL = "qwen/qwen3-next-80b-a3b-thinking"

# System prompt
system_message = (
    "You are a helpful and professional assistant for Intensity, a company providing "
    "cybersecurity, cloud, and AI solutions. "
    "Answer customer questions clearly and accurately. "
    "Keep answers concise, but you may provide more details if the user seems to need them. "
    "Maintain a courteous and friendly tone. "
    "If you don't know the answer, politely admit it and suggest contacting an Intensity expert for further help."
)

# Debug API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")


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
    print(f"Tool get_service_price called for {service_name}")
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

def chat(message, history):
    """
    message: str (new user message)
    history: list of messages in Gradio Chatbot format
    """
    # Convert Gradio history to messages
    messages = [{"role": "system", "content": system_message}]
    
    # Add conversation history
    for entry in history:
        if isinstance(entry, tuple) and len(entry) == 2:
            user_msg, assistant_msg = entry
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
        else:
            # Handle case where history might be in different format
            messages.append({"role": "user", "content": str(entry)})

    # Add new user message
    messages.append({"role": "user", "content": message})
    print("Messages sent to model:", messages)

    try:
        # Call the model
        completion = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = completion.choices[0].message

        # Handle tool calls if any
        if response_message.tool_calls:
            tool_call = response_message.tool_calls[0]
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name == "get_service_price":
                price_result = get_service_price(arguments['service_name'])
                response = f"The price for **{arguments['service_name']}** is {price_result}."
            else:
                response = "I don't know that tool yet."
        else:
            # Normal assistant response
            response = response_message.content

        # Append to history and return
        history.append((message, response))
        return history, ""  # Clear the input box
        
    except Exception as e:
        error_response = f"Sorry, I encountered an error: {str(e)}"
        history.append((message, error_response))
        return history, ""

# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Intensity AI Assistant")
    gr.Markdown("Ask me about Intensity's cybersecurity, cloud, and AI services!")
    
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(
        placeholder="Ask Intensity a question...",
        label="Your Message"
    )
    
    with gr.Row():
        submit = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear Chat")

    # Handle message submission
    submit.click(
        fn=chat,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )
    
    # Allow pressing Enter to send message
    msg.submit(
        fn=chat,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )
    
    # Clear chat history
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(share=False, debug=True)
