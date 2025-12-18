import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
from PIL import Image

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
#             FILE PATHS
# ==========================================

PATH_SHY_BITE          = "memes/pottodu-rakhi.gif"
PATH_ANGRY_FACE        = "memes/pottodu-angry.gif"
PATH_BOTH_HANDS_CHIN   = "memes/pottodu-cries.gif"
PATH_EYEBROWS_UP       = "memes/pottodu-liftingeyebrows.gif"
PATH_BITING_TEETH      = "memes/pottodu-bitingteeth.gif"
PATH_HAND_ON_HEAD      = "memes/ammathodu-pottodu.gif"
PATH_INDEX_ARM_RAISE   = "memes/sai.gif"

# ==========================================
#          SESSION STATE
# ==========================================

if 'gesture_count' not in st.session_state:
    st.session_state.gesture_count = 0
if 'last_gesture' not in st.session_state:
    st.session_state.last_gesture = "None"
if 'current_meme' not in st.session_state:
    st.session_state.current_meme = None

# ==========================================
#          GIF LOADING
# ==========================================

@st.cache_resource
def load_gif_frames(path):
    frames = []
    try:
        gif = Image.open(path)
        frame_index = 0
        while True:
            gif.seek(frame_index)
            frame = gif.convert("RGB")
            frames.append(np.array(frame))
            frame_index += 1
    except EOFError:
        pass
    return frames

@st.cache_resource
def load_all_memes():
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
#         MEDIAPIPE MODELS
# ==========================================

@st.cache_resource
def load_mediapipe_models():
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    
    return face_mesh, hands

# ==========================================
#       GESTURE DETECTION FUNCTIONS
# ==========================================

def detect_shy_gesture(face_lm, hand_lm):
    mouth_center = face_lm[13]
    chin = face_lm[199]
    
    hand_wrist = hand_lm.landmark[0]
    hand_palm = hand_lm.landmark[9]
    hand_index = hand_lm.landmark[8]
    
    for hand_point in [hand_wrist, hand_palm, hand_index]:
        vertical_ok = (hand_point.y > mouth_center.y) and (hand_point.y < chin.y + 0.15)
        horizontal_ok = abs(mouth_center.x - hand_point.x) < 0.15
        if vertical_ok and horizontal_ok:
            return True
    return False

def detect_angry_face(face_lm):
    left_brow_inner = face_lm[55]
    left_brow_outer = face_lm[46]
    right_brow_inner = face_lm[285]
    right_brow_outer = face_lm[276]
    
    left_eye_top = face_lm[159]
    left_eye_bottom = face_lm[145]
    right_eye_top = face_lm[386]
    right_eye_bottom = face_lm[374]
    
    left_brow_avg_y = (left_brow_inner.y + left_brow_outer.y) / 2
    right_brow_avg_y = (right_brow_inner.y + right_brow_outer.y) / 2
    
    left_eye_open = left_eye_bottom.y - left_eye_top.y
    right_eye_open = right_eye_bottom.y - right_eye_top.y
    avg_eye_open = (left_eye_open + right_eye_open) / 2
    
    left_brow_to_eye = left_eye_top.y - left_brow_avg_y
    right_brow_to_eye = right_eye_top.y - right_brow_avg_y
    
    brows_furrowed = (left_brow_to_eye < 0.035) and (right_brow_to_eye < 0.035)
    eyes_narrowed = avg_eye_open < 0.02
    
    return brows_furrowed and eyes_narrowed

def detect_eyebrows_raised(face_lm):
    left_brow_inner = face_lm[55]
    left_brow_outer = face_lm[46]
    right_brow_inner = face_lm[285]
    right_brow_outer = face_lm[276]
    
    left_eye_top = face_lm[159]
    right_eye_top = face_lm[386]
    
    left_brow_avg_y = (left_brow_inner.y + left_brow_outer.y) / 2
    right_brow_avg_y = (right_brow_inner.y + right_brow_outer.y) / 2
    
    left_distance = left_eye_top.y - left_brow_avg_y
    right_distance = right_eye_top.y - right_brow_avg_y
    
    return (left_distance > 0.055) and (right_distance > 0.055)

def detect_biting_teeth(face_lm):
    upper_lip = face_lm[13]
    lower_lip = face_lm[14]
    left_corner = face_lm[61]
    right_corner = face_lm[291]
    
    lip_distance = abs(lower_lip.y - upper_lip.y)
    mouth_width = abs(right_corner.x - left_corner.x)
    
    return lip_distance > 0.015 and mouth_width > 0.08

def detect_both_hands_under_chin(face_lm, hand_list):
    if len(hand_list) < 2:
        return False
    
    chin = face_lm[199]
    hands_under_chin = 0
    
    for hand in hand_list:
        palm = hand.landmark[9]
        wrist = hand.landmark[0]
        
        hand_y = (palm.y + wrist.y) / 2
        hand_x = (palm.x + wrist.x) / 2
        
        vertical_ok = (hand_y >= chin.y - 0.02) and (hand_y < chin.y + 0.12)
        horizontal_ok = abs(hand_x - chin.x) < 0.20
        
        if vertical_ok and horizontal_ok:
            hands_under_chin += 1
    
    return hands_under_chin >= 2

def detect_hand_on_head(face_lm, hand_lm):
    forehead = face_lm[10]
    
    hand_palm = hand_lm.landmark[9]
    hand_wrist = hand_lm.landmark[0]
    hand_middle = hand_lm.landmark[12]
    
    for hand_point in [hand_palm, hand_wrist, hand_middle]:
        above_head = hand_point.y < forehead.y + 0.05
        horizontally_aligned = abs(hand_point.x - forehead.x) < 0.15
        
        if above_head and horizontally_aligned:
            return True
    
    return False

def detect_pointing_at_camera(hand_lm):
    wrist = hand_lm.landmark[0]
    index_mcp = hand_lm.landmark[5]
    index_tip = hand_lm.landmark[8]
    
    middle_tip = hand_lm.landmark[12]
    ring_tip = hand_lm.landmark[16]
    pinky_tip = hand_lm.landmark[20]
    
    index_length_x = abs(index_tip.x - wrist.x)
    base_length_x = abs(index_mcp.x - wrist.x)
    index_extended = index_length_x > base_length_x * 1.3
    
    vertical_diff = abs(index_tip.y - index_mcp.y)
    horizontal_pointing = vertical_diff < 0.08
    
    middle_curled = abs(middle_tip.x - wrist.x) < abs(index_tip.x - wrist.x)
    ring_curled = abs(ring_tip.x - wrist.x) < abs(index_tip.x - wrist.x)
    
    return index_extended and horizontal_pointing and middle_curled and ring_curled

def detect_fist_raised(hand_lm):
    wrist = hand_lm.landmark[0]
    
    index_tip = hand_lm.landmark[8]
    middle_tip = hand_lm.landmark[12]
    ring_tip = hand_lm.landmark[16]
    pinky_tip = hand_lm.landmark[20]
    
    index_mcp = hand_lm.landmark[5]
    middle_mcp = hand_lm.landmark[9]
    ring_mcp = hand_lm.landmark[13]
    pinky_mcp = hand_lm.landmark[17]
    
    index_curled = abs(index_tip.y - wrist.y) < abs(index_mcp.y - wrist.y) + 0.02
    middle_curled = abs(middle_tip.y - wrist.y) < abs(middle_mcp.y - wrist.y) + 0.02
    ring_curled = abs(ring_tip.y - wrist.y) < abs(ring_mcp.y - wrist.y) + 0.02
    pinky_curled = abs(pinky_tip.y - wrist.y) < abs(pinky_mcp.y - wrist.y) + 0.02
    
    all_curled = index_curled and middle_curled and ring_curled and pinky_curled
    arm_raised = wrist.y < 0.6
    
    return all_curled and arm_raised

def evaluate_gesture_priority(shy, angry, eyebrows, teeth, both_hands, head, pointing, fist):
    if fist:
        return "fist_raised", "🤜 Fist Raised", 'arm'
    if pointing:
        return "pointing", "👉 Pointing", None
    if shy:
        return "shy", "😳 Shy Gesture", 'shy'
    if eyebrows:
        return "eyebrows", "😲 Eyebrows Raised", 'eyebrows'
    if head:
        return "head", "🤦 Hand on Head", 'head'
    if both_hands:
        return "both_hands", "🥺 Both Hands Under Chin", 'both_hands'
    if angry:
        return "angry", "😠 Angry Face", 'angry'
    if teeth:
        return "teeth", "😁 Biting Teeth", 'teeth'
    return "none", "None", None

# ==========================================
#       VIDEO TRANSFORMER CLASS
# ==========================================

class GestureDetector(VideoTransformerBase):
    def __init__(self):
        self.face_mesh, self.hands = load_mediapipe_models()
        self.gesture_hold = 0
        self.cooldown_counter = 0
        self.REQUIRED_FRAMES = 6
        self.COOLDOWN_FRAMES = 30
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        face_result = self.face_mesh.process(rgb)
        hands_result = self.hands.process(rgb)
        
        face_lm = None
        hand_lms = []
        
        if face_result.multi_face_landmarks:
            face_lm = face_result.multi_face_landmarks[0].landmark
        
        if hands_result.multi_hand_landmarks:
            hand_lms = hands_result.multi_hand_landmarks
        
        # Detect gestures
        shy = angry = eyebrows = teeth = False
        both_hands = head = pointing = fist = False
        
        if face_lm:
            angry = detect_angry_face(face_lm)
            eyebrows = detect_eyebrows_raised(face_lm)
            teeth = detect_biting_teeth(face_lm)
        
        if hand_lms:
            if len(hand_lms) == 1:
                hl = hand_lms[0]
                pointing = detect_pointing_at_camera(hl)
                fist = detect_fist_raised(hl)
                if face_lm:
                    shy = detect_shy_gesture(face_lm, hl)
                    head = detect_hand_on_head(face_lm, hl)
            
            if len(hand_lms) >= 2 and face_lm:
                both_hands = detect_both_hands_under_chin(face_lm, hand_lms)
        
        gesture_key, gesture_name, meme_key = evaluate_gesture_priority(
            shy, angry, eyebrows, teeth, both_hands, head, pointing, fist
        )
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            gesture_key = "none"
        
        if gesture_key != "none":
            self.gesture_hold += 1
        else:
            self.gesture_hold = 0
        
        # Trigger gesture
        if self.gesture_hold >= self.REQUIRED_FRAMES and self.cooldown_counter == 0:
            self.gesture_hold = 0
            self.cooldown_counter = self.COOLDOWN_FRAMES
            st.session_state.gesture_count += 1
            st.session_state.last_gesture = gesture_name
            if meme_key:
                st.session_state.current_meme = meme_key
        
        # Draw info on frame
        cv2.putText(rgb, f"Gesture: {gesture_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(rgb, f"Count: {st.session_state.gesture_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return rgb

# ==========================================
#            MAIN APP
# ==========================================

def main():
    st.markdown("<h1>🎭 Freak Detector - Cloud Edition</h1>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Statistics")
        st.metric("Total Gestures", st.session_state.gesture_count)
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
        st.markdown("""
        <div class="info-box">
        <strong>💡 Tips:</strong><br>
        • Allow camera access<br>
        • Good lighting helps<br>
        • Position face in center<br>
        • Hold gestures steady
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📹 Live Camera")
        
        # WebRTC configuration
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        webrtc_streamer(
            key="gesture-detection",
            video_transformer_factory=GestureDetector,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )
    
    with col2:
        st.markdown("### 🎬 Meme Response")
        
        if st.session_state.current_meme and st.session_state.current_meme in meme_library:
            frames = meme_library[st.session_state.current_meme]
            if frames:
                # Show first frame
                st.image(frames[0], use_container_width=True)
        else:
            st.info("👀 Waiting for gesture...")

if __name__ == "__main__":
    main()
