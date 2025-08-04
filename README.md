# **The Emotional Intelligence Matrix: A Real-Time Assessment Tool**


This project is a proof-of-concept implementation of "The Emotional Intelligence Matrix," a novel framework designed to assess Emotional Intelligence (EI) in real-time. It combines machine learning for facial emotion recognition with an AI-powered adaptive questioning system to generate a comprehensive personality and emotional profile.


The application, built with Streamlit, uses a webcam to analyze a user's facial expressions, identifies their dominant emotion, and then poses contextually relevant questions to explore their emotional tendencies and personality traits. The final output is a detailed EQ analysis, providing actionable insights for personal growth and professional development.




## 📋 Key Features

* Real-Time Emotion Detection: Utilizes OpenCV and DeepFace to capture and analyze live video from a webcam, identifying emotions like happiness, sadness, anger, and surprise.




* AI-Powered Adaptive Questioning: Employs a GPT-2 language model to dynamically generate workplace-related questions based on the user's detected emotional state.


* Comprehensive EQ Profile: After the session, it calculates an overall EQ score and breaks it down into key dimensions based on established psychological models.

* Interactive Visualizations: Displays the user's EQ profile through an intuitive radar chart and a pie chart for easy interpretation of strengths and areas for development.

* Detailed Analysis & Feedback: Provides AI-generated analysis for each EQ dimension, offering strengths, development areas, and practical improvement strategies.

* Emotion History Logging: Tracks and plots the user's detected emotions over the course of the session.

## 🧠 Theoretical Framework

The system's methodology is grounded in five prominent and widely recognized Emotional Intelligence (EI) models, ensuring a robust and multi-faceted assessment.




* **Goleman's EI Model:** Focuses on self-awareness, self-regulation, motivation, empathy, and social skills as critical competencies for leadership and success.



* **Mayer-Salovey-Caruso EI Model:** Defines EI as a cognitive ability to perceive, understand, and manage emotions effectively.



* **Bar-On's Emotional-Social Intelligence (ESI) Model:** Integrates emotional and social dimensions, including stress management and adaptability, for a holistic view of an individual's functioning.



* **The Six Seconds Model:** Provides a practical framework for applying emotional intelligence in real-world scenarios and everyday life.



* **Plutchik's Wheel of Emotions:** Offers a detailed classification of emotions and their relationships, aiding the system in navigating complex emotional states.


## ⚙️ System Architecture & Technology Stack

The application is built with three core modules that work together to deliver a seamless experience.


* **Facial Recognition and Emotion Detection Module:** Captures the video feed using OpenCV and leverages MediaPipe and a Convolutional Neural Network (CNN) via the DeepFace library to identify facial landmarks and classify emotions.



* **Adaptive Questioning Module:** The detected emotion is passed to a decision-making algorithm that prompts a GPT-2 model to generate a relevant question.


* **Data Fusion and Personality Matrix Module:** User responses and emotion data are synthesized. An algorithm calculates scores for various EI dimensions, which are then used to generate the final report and visualizations.

### Technology Stack

* Programming Language: Python

* Core Libraries:

  * streamlit: For the web application interface.

  * opencv-python: For capturing webcam video feed.

  * deepface: For the core emotion analysis.

  * transformers: For accessing the GPT-2 model for question generation.

  * tensorflow: As a backend for the DeepFace library.

  * matplotlib: For creating the EQ radar and pie charts.

  * numpy: For numerical operations.

## 🚀 Installation and Setup

To run this project locally, follow these steps:

### 1. Clone the repository:


``` git clone https://github.com/YashSingh23/Emotional-Intelligence-Matrix.git ```

``` cd Emotional-Intelligence-Matrix ```

### 2. Create and activate a virtual environment:

Windows:


``` python -m venv venv.\venv\Scripts\activate ```

macOS/Linux: 


``` python3 -m venv venvsource venv/bin/activate ```


### 3. Install the required packages:

Create a ``` requirements.txt ``` file with the following content:

* streamlit

* opencv-python

* deepface

* transformers

* torch

* tensorflow

* matplotlib

* numpy

Then, install the packages using pip:


``` pip install -r requirements.txt ```

### 4. Run the Streamlit application:


``` streamlit run main.py ```

Your web browser should automatically open with the application running.

## 📖 How to Use the Application

**1. Start the Session:** Click the "Start Monitoring" button. Your browser will ask for permission to use your webcam. Please allow it.

**2. Emotion Detection:** Position your face clearly in the camera view. The system will analyze your facial expression and detect your dominant emotion every few seconds.

**3. Answer the Questions:** Based on the detected emotion, a question will appear. Type your thoughtful response in the text box and click "Submit Answer".

**4 Continue the Process:** The system will resume monitoring after you submit your answer. This cycle will repeat.

**5. View Your Report:** When you are finished, click the "Stop Monitoring" button. The session will end, and a comprehensive EQ analysis report will be displayed.

## ✍️ Author

This project is based on the research conducted in the Apex Institute of Technology (CSE), Chandigarh University:


Yash Singh 

