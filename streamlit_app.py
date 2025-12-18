import streamlit as st
import cv2
import numpy as np
import time

# Import all necessary components from freakdetector module
from freakdetector import (
    # Detection functions
    detect_shy_gesture,
    detect_angry_face,
    detect_eyebrows_raised,
    detect_biting_teeth,
    detect_both_hands_under_chin,
    detect_hand_on_head,
    detect_pointing_at_camera,
    detect_fist_raised,
    evaluate_gesture_priority,
    # GIF loading
    load_gif_frames,
    # File paths
    PATH_SHY_BITE,
    PATH_ANGRY_FACE,
    PATH_BOTH_HANDS_CHIN,
    PATH_EYEBROWS_UP,
    PATH_BITING_TEETH,
    PATH_HAND_ON_HEAD,
    PATH_INDEX_POINTING,
    PATH_INDEX_ARM_RAISE,
    # MediaPipe instances
    face_mesh,
    hands
)

# ==========================================
#          PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="🎭 Freak Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
#          CUSTOM CSS STYLING
# ==========================================

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    h1 {
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
        padding: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 10px 0;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    
    .gesture-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    .info-box {
        background: rgba(52, 152, 219, 0.2);
        border-left: 4px solid #3498db;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
#          SESSION STATE INITIALIZATION
# ==========================================

if 'running' not in st.session_state:
    st.session_state.running = False
if 'gesture_count' not in st.session_state:
    st.session_state.gesture_count = 0
if 'last_gesture' not in st.session_state:
    st.session_state.last_gesture = "None"
if 'gesture_history' not in st.session_state:
    st.session_state.gesture_history = []

# ==========================================
#          MEME LOADING
# ==========================================

@st.cache_resource
def load_all_memes():
    """Load all meme GIFs using the imported load_gif_frames function"""
    return {
        'shy': load_gif_frames(PATH_SHY_BITE),
        'angry': load_gif_frames(PATH_ANGRY_FACE),
        'both_hands': load_gif_frames(PATH_BOTH_HANDS_CHIN),
        'eyebrows': load_gif_frames(PATH_EYEBROWS_UP),
        'teeth': load_gif_frames(PATH_BITING_TEETH),
        'head': load_gif_frames(PATH_HAND_ON_HEAD),
        'arm': load_gif_frames(PATH_INDEX_ARM_RAISE),
    }

meme_library = load_all_memes()

# ==========================================
#      GESTURE NAME MAPPING HELPER
# ==========================================

def get_gesture_display_name(gesture_key):
    """Convert gesture key to display name with emoji"""
    gesture_names = {
        "fist_raised": "🤜 Fist Raised",
        "pointing_at_camera": "👉 Pointing at Camera",
        "shy_gesture": "😳 Shy Gesture",
        "eyebrows_raised": "😲 Eyebrows Raised",
        "hand_on_head": "🤦 Hand on Head",
        "both_hands_chin": "🥺 Both Hands Under Chin",
        "angry_face": "😠 Angry Face",
        "biting_teeth": "😁 Biting Teeth",
        "none": "None"
    }
    return gesture_names.get(gesture_key, "None")

# ==========================================
#            MAIN APPLICATION
# ==========================================

def main():
    # Header
    st.markdown("<h1>🎭 Freak Detector - Real-Time Gesture Recognition</h1>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")
        
        if st.button("🎥 Start Detection" if not st.session_state.running else "⏹️ Stop Detection"):
            st.session_state.running = not st.session_state.running
        
        st.markdown("---")
        
        st.markdown("### 📊 Statistics")
        st.metric("Total Gestures Detected", st.session_state.gesture_count)
        st.metric("Last Gesture", st.session_state.last_gesture)
        
        st.markdown("---")
        
        st.markdown("### 🎭 Supported Gestures")
        gestures = [
            "🤜 Fist Raised",
            "👉 Pointing at Camera",
            "😳 Shy Gesture",
            "🤦 Hand on Head",
            "😠 Angry Face",
            "😲 Eyebrows Raised",
            "😁 Biting Teeth",
            "🥺 Both Hands Under Chin"
        ]
        for gesture in gestures:
            st.markdown(f'<div class="gesture-badge">{gesture}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### ⚡ Settings")
        sensitivity = st.slider("Detection Sensitivity", 1, 10, 6, 
                               help="Lower = more sensitive")
        cooldown = st.slider("Cooldown (frames)", 10, 60, 30,
                            help="Pause between gesture detections")
        
        st.markdown("---")
        
        st.markdown("""
        <div class="info-box">
        <strong>💡 Tips:</strong><br>
        • Hold gestures for ~200ms<br>
        • Good lighting helps<br>
        • Position face in center<br>
        • Press ESC to exit
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📹 Live Webcam Feed")
        webcam_placeholder = st.empty()
    
    with col2:
        st.markdown("### 🎬 Meme Response")
        meme_placeholder = st.empty()
    
    status_placeholder = st.empty()
    
    # Video processing
    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        
        gesture_hold = 0
        cooldown_counter = 0
        current_meme_frames = []
        current_meme_index = 0
        
        REQUIRED_FRAMES = sensitivity
        COOLDOWN_FRAMES = cooldown
        
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe (imported from freakdetector)
            face_result = face_mesh.process(rgb)
            hands_result = hands.process(rgb)
            
            face_lm = None
            hand_lms = []
            
            if face_result.multi_face_landmarks:
                face_lm = face_result.multi_face_landmarks[0].landmark
            
            if hands_result.multi_hand_landmarks:
                hand_lms = hands_result.multi_hand_landmarks
            
            # Detect gestures using imported functions
            shy_gesture = False
            both_hands_chin = False
            hand_on_head = False
            pointing_at_camera = False
            fist_raised = False
            angry_face = False
            eyebrows_raised = False
            biting_teeth = False
            
            if face_lm:
                angry_face = detect_angry_face(face_lm)
                eyebrows_raised = detect_eyebrows_raised(face_lm)
                biting_teeth = detect_biting_teeth(face_lm)
            
            if hand_lms:
                if len(hand_lms) == 1:
                    hl = hand_lms[0]
                    pointing_at_camera = detect_pointing_at_camera(hl)
                    fist_raised = detect_fist_raised(hl)
                    if face_lm:
                        shy_gesture = detect_shy_gesture(face_lm, hl)
                        hand_on_head = detect_hand_on_head(face_lm, hl)
                
                if len(hand_lms) >= 2 and face_lm:
                    both_hands_chin = detect_both_hands_under_chin(face_lm, hand_lms)
            
            # Evaluate gesture priority using imported function
            gesture_key, gesture_name = evaluate_gesture_priority(
                shy_gesture, angry_face, eyebrows_raised, biting_teeth,
                both_hands_chin, hand_on_head, pointing_at_camera, fist_raised
            )
            
            if cooldown_counter > 0:
                cooldown_counter -= 1
                gesture_key = "none"
            
            if gesture_key != "none":
                gesture_hold += 1
            else:
                gesture_hold = 0
            
            # Trigger meme
            if gesture_hold >= REQUIRED_FRAMES and cooldown_counter == 0:
                gesture_hold = 0
                cooldown_counter = COOLDOWN_FRAMES
                st.session_state.gesture_count += 1
                st.session_state.last_gesture = gesture_name
                st.session_state.gesture_history.append(gesture_name)
                
                # Load appropriate meme
                if gesture_key == "fist_raised":
                    current_meme_frames = meme_library['arm']
                elif gesture_key == "shy_gesture":
                    current_meme_frames = meme_library['shy']
                elif gesture_key == "hand_on_head":
                    current_meme_frames = meme_library['head']
                elif gesture_key == "angry_face":
                    current_meme_frames = meme_library['angry']
                elif gesture_key == "eyebrows_raised":
                    current_meme_frames = meme_library['eyebrows']
                elif gesture_key == "biting_teeth":
                    current_meme_frames = meme_library['teeth']
                elif gesture_key == "both_hands_chin":
                    current_meme_frames = meme_library['both_hands']
                
                current_meme_index = 0
            
            # Display webcam
            webcam_placeholder.image(rgb, channels="RGB", use_container_width=True)
            
            # Display meme
            if len(current_meme_frames) > 0:
                meme_frame = current_meme_frames[current_meme_index]
                meme_placeholder.image(meme_frame, channels="RGB", use_container_width=True)
                current_meme_index = (current_meme_index + 1) % len(current_meme_frames)
            else:
                meme_placeholder.info("👀 Waiting for gesture...")
            
            # Status
            status_text = f"🎯 Status: **{'Detecting...' if gesture_key == 'none' else gesture_name}** | Cooldown: {cooldown_counter}"
            status_placeholder.markdown(status_text)
            
            time.sleep(0.03)
        
        cap.release()
    else:
        st.info("👆 Click 'Start Detection' in the sidebar to begin!")

if __name__ == "__main__":
    main()
