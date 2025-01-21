# importing libraries
from transformers import pipeline
import time
import gradio as gr

# creating pipeline for importing model from Hugging Face
pipe = pipeline("text-generation", model="openai-community/gpt2", )

# Function to process user input and generate a response
def chatbot_response(user_input):
    responses = pipe(
        user_input,
        max_length = 100,
        min_length = 20,
        temperature = 0.7
    )
    full_response = responses[0]["generated_text"]

    # Simulating slow generation word by word
    slow_response = ""
    for word in full_response.split():
        slow_response += word + " "
        time.sleep(0.3)  # Adjust the delay (e.g., 0.3 seconds per word)
    
    return slow_response

# creating gradio interface

chatbot_ui = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(
        lines=3,
        max_lines=5,
        placeholder="Type your message here...",
        label="Ubi Asks"
    ),
    outputs=gr.Textbox(
        label="Ubi Talks",
        lines=5,
        max_lines=10,
    ),
    title="Ubi Talks",
    description="Welcome to Ubi Talks! A chatbot powered by GPT-2. Enjoy a conversational experience where responses are generated dynamically!",
    layout="vertical",
    allow_flagging="never"
)

# Launch the gradio app
if __name__ == "__main__":
    try:
        chatbot_ui.launch(share=False,debug=True)
    except Exception as e:
        print(f"Error launching app: {str(e)}")