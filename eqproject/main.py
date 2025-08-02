# --- Streamlit Config MUST BE FIRST ---
import streamlit as st

st.set_page_config(page_title="Emotion-Driven Q&A", layout="wide")

# --- Other Imports AFTER Config ---
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from deepface import DeepFace
from transformers import pipeline


# --- Cached Resources ---
@st.cache_resource
def load_llm():
    return pipeline("text-generation", model="gpt2")


llm = load_llm()


# --- Helper Functions ---
def generate_question(emotion):
    """Generate a complete workplace-related question based on detected emotion."""
    prompt = f"""Generate a specific workplace-related question about handling {emotion} in a professional setting. 
    The question should be complete and encourage reflection. Example: "How do you maintain focus when feeling overwhelmed by multiple projects?"
    Question:"""

    response = llm(
        prompt,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        truncation=True
    )

    # Clean up the response and ensure it's a complete question
    full_question = response[0]['generated_text'].replace(prompt, "").strip()
    if full_question.endswith('?'):
        return full_question
    # Add question mark if missing and take first sentence
    return full_question.split('.')[0].split('?')[0] + '?'


def plot_emotion_history(emotion_log):
    """Plot detected emotions over time."""
    if not emotion_log:
        st.warning("No emotion data to display.")
        return

    times = [entry[0] for entry in emotion_log]
    emotions = [entry[1] for entry in emotion_log]

    plt.figure(figsize=(10, 4))
    plt.plot(times, emotions, marker='o')
    plt.title("Emotion Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Emotion")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)


def calculate_eq_score(user_responses):
    """Calculate Advanced EQ Score based on user responses."""
    if not user_responses:
        return 0, {}

    # Initialize EQ dimensions
    eq_dimensions = {
        "Self-Awareness": 0,
        "Self-Regulation": 0,
        "Social Awareness": 0,
        "Relationship Management": 0,
        "Resilience": 0
    }

    # Emotion to EQ dimension mapping (based on Plutchik’s Wheel)
    emotion_to_dimension = {
        "happy": ["Self-Awareness", "Relationship Management"],
        "sad": ["Resilience", "Self-Regulation"],
        "angry": ["Self-Regulation", "Resilience"],
        "fear": ["Self-Regulation", "Social Awareness"],
        "surprise": ["Self-Awareness", "Social Awareness"],
        "disgust": ["Self-Regulation", "Resilience"],
        "trust": ["Relationship Management", "Social Awareness"],
        "anticipation": ["Self-Awareness", "Resilience"]
    }

    # Calculate scores
    for response in user_responses:
        emotion = response['emotion'].lower()
        if emotion in emotion_to_dimension:
            for dimension in emotion_to_dimension[emotion]:
                eq_dimensions[dimension] += 5  # Assign 5 points per response

    # Calculate total EQ score (out of 100)
    eq_score = sum(eq_dimensions.values())
    return eq_score, eq_dimensions


def plot_eq_radar(eq_dimensions):
    """Plot radar chart of EQ dimensions."""
    labels = list(eq_dimensions.keys())
    scores = list(eq_dimensions.values())

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    scores += scores[:1]  # Close the radar chart
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, color='skyblue', alpha=0.6)
    ax.plot(angles, scores, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("EQ Dimensions Radar Chart", size=14, color='blue', y=1.1)
    return fig


def generate_dimension_analysis(dimension, score, responses):
    """Generate detailed analysis for a single EQ dimension."""
    prompt = f"""Analyze the {dimension} capability (score {score}/20) of a professional. 
    Consider these workplace responses: {[r['answer'] for r in responses]}
    Provide:
    1. Key strength highlighted by score
    2. Main development area
    3. Practical improvement strategy
    Relate analysis to responses. Use concise, professional language (max 120 words)."""

    return llm(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']


def display_eq_summary(eq_dimensions, user_responses):
    """Display detailed EQ dimension analysis with visualizations."""
    st.markdown("## 📊 Comprehensive EQ Analysis Report")

    # Main layout columns
    col_viz, col_summary = st.columns([1, 2])

    with col_viz:
        st.markdown("### 📈 EQ Visualization")
        st.pyplot(plot_eq_radar(eq_dimensions))

        # Pie Chart
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            eq_dimensions.values(),
            labels=eq_dimensions.keys(),
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.Pastel1.colors
        )
        ax.set_title("EQ Dimension Distribution")
        st.pyplot(fig)

    with col_summary:
        st.markdown("### 📝 Dimension Breakdown")
        dimension_descriptions = {
            "Self-Awareness": "Understanding personal emotions and their workplace impact",
            "Self-Regulation": "Managing emotional responses in professional situations",
            "Social Awareness": "Recognizing and understanding colleagues' emotions",
            "Relationship Management": "Building effective professional relationships",
            "Resilience": "Adapting to challenges and recovering from setbacks"
        }

        for dim, score in eq_dimensions.items():
            with st.expander(f"{dim} ({score}/20)", expanded=True):
                st.caption(f"**Definition:** {dimension_descriptions[dim]}")
                analysis = generate_dimension_analysis(dim, score, user_responses)
                st.write(analysis)
                st.progress(score / 20)


# --- Initialize Session State ---
session_defaults = {
    "emotion_log": [],
    "monitoring_active": False,
    "awaiting_response": False,
    "current_question": "",
    "current_emotion": "",
    "user_responses": [],
    "last_detection_time": time.time(),
    "camera": None,
    "show_summary": False
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Page Content ---
st.title("🎥 Real-Time Emotion Monitoring & Q&A")
st.write("The system detects your emotions, asks related questions, and waits for your response before continuing.")

# --- Start/Stop Buttons ---
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Monitoring") and not st.session_state.monitoring_active:
        st.session_state.monitoring_active = True
        st.session_state.camera = cv2.VideoCapture(0)

with col2:
    if st.button("Stop Monitoring") and st.session_state.monitoring_active:
        st.session_state.monitoring_active = False
        st.session_state.show_summary = True
        if st.session_state.camera and st.session_state.camera.isOpened():
            st.session_state.camera.release()
        st.rerun()

# --- Main Processing Loop ---
if st.session_state.monitoring_active:
    FRAME_WINDOW = st.empty()

    try:
        if not st.session_state.camera.isOpened():
            st.error("Camera connection lost. Please restart monitoring.")
            st.session_state.monitoring_active = False
        else:
            ret, frame = st.session_state.camera.read()
            if ret:
                # Mirror the frame for natural experience
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(rgb_frame)

                # Process emotion every 5 seconds if not waiting for response
                if not st.session_state.awaiting_response:
                    current_time = time.time()
                    if current_time - st.session_state.last_detection_time >= 5:
                        try:
                            result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
                            dominant_emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result[
                                'dominant_emotion']
                            timestamp = time.strftime("%H:%M:%S", time.localtime())

                            st.session_state.emotion_log.append((timestamp, dominant_emotion))
                            st.session_state.current_emotion = dominant_emotion
                            st.success(f"Detected Emotion: **{dominant_emotion}** at {timestamp}")

                            question = generate_question(dominant_emotion)
                            st.session_state.current_question = question
                            st.session_state.awaiting_response = True
                            st.session_state.last_detection_time = current_time

                        except Exception as e:
                            st.error(f"Error analyzing frame: {str(e)}")
                            st.session_state.last_detection_time = current_time

    except Exception as e:
        st.error(f"Camera error: {str(e)}")
        st.session_state.monitoring_active = False
        if st.session_state.camera and st.session_state.camera.isOpened():
            st.session_state.camera.release()

# --- Question & Response Section ---
if st.session_state.awaiting_response:
    st.markdown(f"### 🤔 **Question:** {st.session_state.current_question}")
    user_input = st.text_input("Your Answer:", key="user_answer")

    if st.button("Submit Answer"):
        if user_input.strip():
            st.session_state.user_responses.append({
                "emotion": st.session_state.current_emotion,
                "question": st.session_state.current_question,
                "answer": user_input
            })
            st.success("✅ Answer submitted! Resuming monitoring...")
            st.session_state.awaiting_response = False
            st.session_state.current_question = ""

            # Clear input field and force UI refresh
            if 'user_answer' in st.session_state:
                del st.session_state.user_answer
            st.rerun()
        else:
            st.warning("⚠️ Please provide an answer before proceeding.")

# --- Post-Monitoring Analysis ---
if not st.session_state.monitoring_active and st.session_state.show_summary:
    # Calculate EQ scores
    eq_score, eq_dimensions = calculate_eq_score(st.session_state.user_responses)

    # Display comprehensive report
    display_eq_summary(eq_dimensions, st.session_state.user_responses)

    # Raw Data Section
    with st.expander("📁 View Raw Data"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Emotion Log")
            st.write(st.session_state.emotion_log)
        with col2:
            st.write("### User Responses")
            st.write(st.session_state.user_responses)

# --- Emotion History Plot ---
if st.session_state.emotion_log and st.session_state.monitoring_active:
    st.markdown("### 📊 Emotion History Over Time")
    plot_emotion_history(st.session_state.emotion_log)

# --- Display User Responses ---
if st.session_state.user_responses and st.session_state.monitoring_active:
    st.markdown("### 📖 Your Responses")
    for idx, response in enumerate(st.session_state.user_responses, 1):
        st.write(f"**{idx}. Emotion:** {response['emotion']}")
        st.write(f"**Question:** {response['question']}")
        st.write(f"**Answer:** {response['answer']}")
        st.markdown("---")

    if st.button("Clear Responses"):
        st.session_state.user_responses = []
        st.rerun()

# --- Cleanup on App Exit ---
if not st.session_state.monitoring_active and st.session_state.camera:
    st.session_state.camera.release()