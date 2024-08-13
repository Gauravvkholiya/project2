import streamlit as st
import speech_recognition as sr
from ultralytics import YOLO
import nltk
import pyttsx3
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import namedtuple
import cv2
# from streamlit_toggle import st_toggle_switch
import serial

# Initialize model
model = YOLO("yolov8n.pt")

# Streamlit state for detected objects
if 'detected_objects' not in st.session_state:
    st.session_state.detected_objects = {}

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        text = ""
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"You said: {text}")
        except sr.UnknownValueError:
            st.write("Sorry, I did not understand that.")
        except sr.RequestError:
            st.write("Sorry, my speech service is down.")
    return text

def process_question(question):
    words = word_tokenize(question)
    words = [word for word in words if word.isalpha()]
    words = [word.lower() for word in words if word not in stopwords.words('english')]
    return words

# Define a named tuple for pairs
Pair = namedtuple('Pair', ['word', 'value'])

def query_json(data, words):
    results = []  # List to store the results
    for key, value in data.items():
        for word in words:
            if word.lower() in key.lower():
                results.append(Pair(word, value))
                break  # Stop searching once a match is found
    if results:
        return results
    return "Sorry, I couldn't find the answer."

def speak_text(text):
    engine = pyttsx3.init()
    if not text:
        engine.say("Not found")
    else:
        text_to_be_spoken = ' '.join([f"There are {count} {word}" for word, count in text])
        engine.say(text_to_be_spoken)
    engine.runAndWait()



def capture_and_display():
    # Open the webcam (0 is the default camera)
    cap = cv2.VideoCapture("http://192.168.137.14:81/stream")

    if not cap.isOpened():
        st.write("Error: Could not open webcam.")
        return {}

    # Dictionary to store detected objects and their counts
    detected_objects = {}
    
    #DISTANCE
    ser = serial.Serial('COM7', 115200)


    duration = 200  # seconds
    start_time = time.time()
    toggle = 0

    # Placeholder for the image
    image_placeholder = st.empty()
    text_placeholder = st.empty()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            st.write("Error: Failed to capture image.")
            break
             
        results = model.predict(source=frame, device='cpu')

            # Process detection results
        for result in results:
            for box in result.boxes:
                class_id = box.cls.item()
                class_name = model.names[class_id]
                if class_name not in detected_objects:
                    detected_objects[class_name] = 0
                detected_objects[class_name] += 1

        # Get the resulting frame with bounding boxes
        annotated_frame = results[0].plot()

        # Convert annotated frame to a format Streamlit can display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Update the placeholder with the new frame
        image_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)

        if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').rstrip()
                text_placeholder.write(line)
        # Check if the duration has passed
        if time.time() - start_time > duration:
            break
        
    cap.release()
    st.session_state.detected_objects = detected_objects
    return detected_objects

# Streamlit UI
st.title("Real-time Object Detection and Speech Interaction")

# if st.button("toggle"):
#     toggle = 1;
#     st.write("Detected objects after recording:", st.session_state.detected_objects)
if st.button("Start Recording"):
    st.write("Recording and detecting objects...")
    detected_objects = capture_and_display()
    st.write("Detected objects after recording:", st.session_state.detected_objects)

if st.button("ASK"):
    st.write("Listening, speak after 2 seconds...")
    time.sleep(2)  # Allow time for user to prepare
    text = recognize_speech()
    st.write("You said:", text)
    words = process_question(text)
    st.write("Processed words:", words)
    ans = query_json(st.session_state.detected_objects, words)
    speak_text(ans)
    st.write("Answer:", ans)