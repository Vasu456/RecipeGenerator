import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import google.generativeai as genai
import os 
from dotenv import load_dotenv

load_dotenv()
# Configure Genai Key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize Streamlit app
st.set_page_config(page_title="Recipe Generator App", page_icon="🍔", layout="centered")
st.header("Recipe Generator")



model_chat = genai.GenerativeModel("gemini-pro")
chat = model_chat.start_chat(history=[])
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
# Function to get a response from Gemini with the specified context and question
def get_gemini_response(prompt):
    response = chat.send_message(prompt)
    return response
# Function to get an answer to the user's question about the recipe
def get_answer(question, recipe):
    # Create a prompt to get an answer to the question about the recipe
    prompt = f"Here is a recipe: {recipe}\n\n{question}. Answer in the friendly manner. The response should not contain ** at the beginning and end and also do not bold any text"
    response = get_gemini_response(prompt)
    return response.text
            
def get_gemini_response_for_receipe(question, prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([prompt, question])
    return response.text

# Load the pre-trained model
model_apna = load_model('my_model.h5')

# Define class labels
class_labels = ['Biryani', 'Burger', 'Dosa', 'Idly', 'Pizza']

def preprocess_image(image):
    resized_image = cv2.resize(image, (150, 150))
    normalized_image = resized_image / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)
    return input_image

def predict_class(image):
    input_image = preprocess_image(image)
    predictions = model_apna.predict(input_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label

def get_recipe(dish_name):
    prompt = f"Please provide a recipe for {dish_name}."
    response = get_gemini_response_for_receipe(dish_name, prompt)
    return response

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
# hard coding the image for now
# uploaded_image = open("1.jpg", "rb")
if uploaded_image is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
    
    # # Convert BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(rgb_image, caption='Uploaded Image', width=300)
    predicted_class = predict_class(image)
    st.success(f'The predicted recipe category is: {predicted_class}')
    # Get the recipe for the predicted dish category
    recipe = get_recipe(predicted_class)
    if recipe:
        st.subheader("Recipe:")
        st.write(recipe)
        user_question = st.text_input("You: ")
        if st.button("Ask"):
            # st.write("User question:", user_question)  # Check user question
            # st.write("Recipe:", recipe)  # Check recipe
            response = get_answer(user_question, recipe)
            if response:
                st.write(f"AI : {response}")
            else:
                st.write("Gemini response is empty")

# if uploaded_image is not None:
#     image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
    
#     # Convert BGR image to RGB
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     st.image(rgb_image, caption='Uploaded Image', width=300)

#     if st.button('Predict'):
#         predicted_class = predict_class(image)
#         st.success(f'The predicted recipe category is: {predicted_class}')

#         # Get the recipe for the predicted dish category
#         recipe = get_recipe(predicted_class)
#         if recipe:
#             st.subheader("Recipe:")
#             st.write(recipe)
#             user_question = st.text_input("You: ")
#             if user_question.strip() != "" and st.button("Ask"):
#                 print("User question:", user_question)  # Check user question
#                 print("Recipe:", recipe)  # Check recipe
#                 response = get_answer(user_question, recipe)
#                 st.write(f"AI : {response}")
#                 print("Gemini response:", response)  # Check Gemini response