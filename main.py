import streamlit as st
import cv2
import numpy as np
from paddleocr import PaddleOCR
from groq import Groq
from dotenv import load_dotenv
import os

# Initialize OCR readers
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')


# Load environment variables from .env file
load_dotenv()

# Get the API key from the .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# Function to preprocess image
def preprocess_image(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    return img

# Function to extract text using PaddleOCR
def extract_text_paddle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = paddle_ocr.ocr(gray, cls=True)
    extracted_text = " ".join([line[1][0] for result in results for line in result])
    return extracted_text



# Function to generate structured output using Groq
def generate_structured_output(text):
    template = f"""
    Extract the following details from the provided text:
    1. Company Name
    2. Designation
    3. Phone Number
    4. Address
    5. Email ID
    6. Web links

    Text:
    {text}
    """
    prompt = template.format(text=f"{text}")
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192"
    )
    answer = chat_completion.choices[0].message.content
    return answer

st.title("Business Card Text Extraction-1")

image_input = st.camera_input("Capture a business card")

if image_input:
    # Preprocess image
    image = preprocess_image(image_input.read())
    
    if st.button("Extract"):
        extracted_text = extract_text_paddle(image)
    
    # Display structured output
    if 'extracted_text' in locals():
        structured_output = generate_structured_output(extracted_text)
        st.subheader("Available details in Business card")
        st.text(structured_output)
