import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Set page config
st.set_page_config(
    page_title="Animal Classification App",
    page_icon="🐾",
    layout="wide"
)

# Title and description
st.title("🐾 Real-time Animal Classification")
st.write("This app uses your webcam or uploaded images to classify animals.")

# Load the model
@st.cache_resource
def load_animal_model():
    try:
        model = load_model('model.h5')
        return model
    except:
        st.error("Model file not found. Please make sure 'model.h5' exists in the current directory.")
        return None

# Function to preprocess frame
def preprocess_frame(frame):
    # Resize frame to match model input size
    frame = cv2.resize(frame, (224, 224))
    # Convert to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Normalize
    frame = frame / 255.0
    # Add batch dimension
    frame = np.expand_dims(frame, axis=0)
    return frame

# Function to get class names
def get_class_names():
    try:
        # Get class names from the dataset directory
        train_dir = os.path.join('animaldaset', 'train')
        if not os.path.exists(train_dir):
            st.error(f"Training directory not found at {train_dir}")
            return []
            
        class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        if not class_names:
            st.error("No class folders found in training directory")
            return []
            
        st.success(f"Found {len(class_names)} animal classes")
        return class_names
    except Exception as e:
        st.error(f"Error loading class names: {str(e)}")
        return []

# Function to make predictions
def predict_image(img, model, class_names):
    if model is None or not class_names:
        return None, None, None
    
    try:
        # Preprocess the image
        processed_img = preprocess_frame(img)
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [(class_names[idx], float(predictions[0][idx]) * 100) for idx in top_3_idx]
        
        return class_names[np.argmax(predictions[0])], float(np.max(predictions[0]) * 100), top_3_predictions
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

# Load model and class names
model = load_animal_model()
class_names = get_class_names()

if not class_names:
    st.error("Please make sure your dataset is properly organized in the animaldaset/train directory")
    st.stop()

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📸 Webcam", "📤 Upload Image"])

with tab1:
    st.write("### Camera Feed")
    # Camera input
    camera_input = st.camera_input("Take a picture")
    
    if camera_input is not None:
        # Convert the image to bytes
        bytes_data = camera_input.getvalue()
        # Convert bytes to numpy array
        nparr = np.frombuffer(bytes_data, np.uint8)
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Make prediction
        predicted_class, confidence, top_3 = predict_image(frame, model, class_names)
        
        if predicted_class:
            st.write("### Classification Results")
            st.write(f"**Predicted Animal:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            st.write("**Top 3 Predictions:**")
            for animal, conf in top_3:
                st.write(f"- {animal}: {conf:.2f}%")

with tab2:
    st.write("### Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Make prediction
        predicted_class, confidence, top_3 = predict_image(img, model, class_names)
        
        if predicted_class:
            st.write("### Classification Results")
            st.write(f"**Predicted Animal:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            st.write("**Top 3 Predictions:**")
            for animal, conf in top_3:
                st.write(f"- {animal}: {conf:.2f}%")

# Add some information about the app
st.markdown("---")
st.write("""
### How to use:
1. Choose between webcam or image upload
2. For webcam:
   - Click the 'Start Camera' button
   - Take a picture using the camera
3. For image upload:
   - Click 'Browse files' to select an image
   - Supported formats: JPG, JPEG, PNG
4. The app will automatically classify the animal in the image
5. Results will show the predicted animal and confidence score

### Note:
- Make sure you have good lighting
- Keep the animal clearly visible in the frame
- The model works best with clear, front-facing images of animals
""") 