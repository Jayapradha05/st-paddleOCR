PaddleOCR Business Card Extraction
Overview
This application uses PaddleOCR to extract information from business cards. It captures an image using a camera, extracts text from the card, and structures the information into fields such as Name, Phone Number, Address, and Email ID.

Features
Capture an image using the camera.
Extract text from the business card using PaddleOCR.
Structure the extracted text into meaningful fields.
User-friendly Streamlit interface.
Requirements
Python version: 3.8 or higher

Libraries:

streamlit
opencv-python
numpy
paddleocr
groq
python-dotenv
Environment Variables:

Create a .env file in the root directory with the following content:
makefile
Copy code
GROQ_API_KEY=<your_groq_api_key>
Installation
Clone the repository:

bash
Copy code
git clone <repository_url>
cd paddleocr-business-card-extraction
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the application:

bash
Copy code
streamlit run app.py
Usage
Launch the app with streamlit run app.py.
Use the camera interface to capture an image of the business card.
Click the Submit button to extract and display structured information.
Code Workflow
Image Capture: The app captures the image using st.camera_input.
Preprocessing: Converts the image into a NumPy array for processing.
Text Extraction: Utilizes PaddleOCR for recognizing text.
Structuring Output: Calls Groq API to organize the text into key-value pairs.
Display: Outputs structured fields in the app interface.
Troubleshooting
PaddleOCR not found: Ensure PaddleOCR is installed and paddleocr is imported.
Camera not working: Ensure camera permissions are enabled for Streamlit.
Invalid API Key: Double-check the Groq API key in the .env file.