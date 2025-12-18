import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import threading
import queue
import time
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
#             FILE PATHS (ALL MEMES)
# ==========================================

PATH_SHY_BITE          = "memes/pottodu-rakhi.gif"
PATH_ANGRY_FACE        = "memes/pottodu-angry.gif"
PATH_BOTH_HANDS_CHIN   = "memes/pottodu-cries.gif"
PATH_EYEBROWS_UP       = "memes/pottodu-liftingeyebrows.gif"
PATH_BITING_TEETH      = "memes/pottodu-bitingteeth.gif"
PATH_HAND_ON_HEAD      = "memes/ammathodu-pottodu.gif"
PATH_INDEX_POINTING     = "memes/thaali-pottodu-converted.mp4"
PATH_INDEX_ARM_RAISE    = "memes/sai.gif"

# ==========================================
#             GLOBAL SETTINGS
# ==========================================

IDLE_MIN_FRAMES = 70
COOLDOWN_FRAMES = 30
REQUIRED_FRAMES = 6

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

idle_hold = 0
palm_dist_buffer = deque(maxlen=20)

# ==========================================
#          GIF LOADING ENGINE
# ==========================================

@st.cache_resource
def load_gif_frames(path):
    """Load all GIF frames as a list of BGR numpy arrays."""
    frames = []
    try:
        gif = Image.open(path)
    except:
        print(f"❌ Error loading GIF: {path}")
        return frames

    try:
        frame_index = 0
        while True:
            gif.seek(frame_index)
            frame = gif.convert("RGB")
            np_frame = np.array(frame)
            bgr = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
            frames.append(bgr)
            frame_index += 1
    except EOFError:
        pass

    return frames


# Preload all GIFs at start for speed
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
#       IMAGE LOADING FOR STILL MEMES
# ==========================================

# No static images currently used

# ==========================================
#         MEDIAPIPE INITIALIZATION
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

face_mesh, hands = load_mediapipe_models()

# ==========================================
#            VIDEO PLAYBACK (NOT USED IN STREAMLIT)
# ==========================================

video_queue = queue.Queue(maxsize=1)
play_video_flag = threading.Event()
video_done_flag = threading.Event()

CURRENT_VIDEO = None


def video_player_thread():
    """Thread that plays MP4 meme videos ONCE and outputs frames to queue."""
    global CURRENT_VIDEO

    while True:
        play_video_flag.wait()

        vid = cv2.VideoCapture(CURRENT_VIDEO)

        if not vid.isOpened():
            print("❌ Unable to open video:", CURRENT_VIDEO)
            play_video_flag.clear()
            continue

        while play_video_flag.is_set():
            ok, frame = vid.read()
            if not ok:
                break

            if not video_queue.full():
                video_queue.put(frame)

            time.sleep(0.01)

        vid.release()
        video_done_flag.set()
        play_video_flag.clear()


threading.Thread(target=video_player_thread, daemon=True).start()


# ==========================================
#      STATE MANAGEMENT + GLOBAL SETTINGS
# ==========================================

COOLDOWN_FRAMES = 30
cooldown = 0
REQUIRED_FRAMES = 6

nose_x_buffer = deque(maxlen=15)
nose_y_buffer = deque(maxlen=20)

active_mode = "idle"
current_gif_frames = []
current_gif_index = 0

# ==========================================
#         MEDIAPIPE INITIALIZATION
# ==========================================

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ==========================================
#       GESTURE DETECTION FUNCTIONS
# ==========================================

def detect_shy_gesture(face_lm, hand_lm):
    mouth_center = face_lm[13]
    lower_lip = face_lm[14]
    chin = face_lm[199]
    
    hand_wrist = hand_lm.landmark[0]
    hand_palm = hand_lm.landmark[9]
    hand_index = hand_lm.landmark[8]
    hand_thumb = hand_lm.landmark[4]
    
    for hand_point in [hand_wrist, hand_palm, hand_index, hand_thumb]:
        vertical_ok = (hand_point.y > mouth_center.y) and (hand_point.y < chin.y + 0.15)
        dx = mouth_center.x - hand_point.x
        horizontal_ok = abs(dx) < 0.15
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
    
    showing_teeth = lip_distance > 0.015 and mouth_width > 0.08
    return showing_teeth


def detect_both_hands_under_chin(face_lm, hand_list):
    """
    Detects both hands placed under chin
    """
    if len(hand_list) < 2:
        return False
    
    chin = face_lm[199]
    chin_y = chin.y
    chin_x = chin.x
    
    hands_under_chin = 0
    
    for hand in hand_list:
        palm = hand.landmark[9]
        wrist = hand.landmark[0]
        
        hand_y = (palm.y + wrist.y) / 2
        hand_x = (palm.x + wrist.x) / 2
        
        vertical_ok = (hand_y >= chin_y - 0.02) and (hand_y < chin_y + 0.12)
        horizontal_ok = abs(hand_x - chin_x) < 0.20
        
        if vertical_ok and horizontal_ok:
            hands_under_chin += 1
    
    return hands_under_chin >= 2


def detect_hand_on_head(face_lm, hand_lm):
    """
    Detects hand placed on top of head
    """
    forehead = face_lm[10]  # Top of forehead
    
    # Check multiple hand points
    hand_palm = hand_lm.landmark[9]
    hand_wrist = hand_lm.landmark[0]
    hand_middle = hand_lm.landmark[12]
    
    # Hand should be above forehead
    for hand_point in [hand_palm, hand_wrist, hand_middle]:
        above_head = hand_point.y < forehead.y + 0.05
        horizontally_aligned = abs(hand_point.x - forehead.x) < 0.15
        
        if above_head and horizontally_aligned:
            return True
    
    return False


def detect_pointing_at_camera(hand_lm):
    """
    Detects index finger pointing horizontally towards camera
    Index finger extended forward, other fingers curled
    """
    # Get key landmarks
    wrist = hand_lm.landmark[0]
    index_mcp = hand_lm.landmark[5]   # Index finger base
    index_pip = hand_lm.landmark[6]   # Index finger middle joint
    index_tip = hand_lm.landmark[8]   # Index finger tip
    
    middle_mcp = hand_lm.landmark[9]
    middle_tip = hand_lm.landmark[12]
    
    ring_mcp = hand_lm.landmark[13]
    ring_tip = hand_lm.landmark[16]
    
    pinky_mcp = hand_lm.landmark[17]
    pinky_tip = hand_lm.landmark[20]
    
    thumb_tip = hand_lm.landmark[4]
    
    # Check if index finger is extended (tip further from wrist than base)
    index_length_x = abs(index_tip.x - wrist.x)
    base_length_x = abs(index_mcp.x - wrist.x)
    index_extended = index_length_x > base_length_x * 1.3
    
    # Check if index finger is relatively horizontal (not pointing up or down too much)
    # Y difference between tip and base should be small
    vertical_diff = abs(index_tip.y - index_mcp.y)
    horizontal_pointing = vertical_diff < 0.08
    
    # Other fingers should be curled (closer to palm)
    middle_curled = abs(middle_tip.x - wrist.x) < abs(middle_mcp.x - wrist.x) * 1.2
    ring_curled = abs(ring_tip.x - wrist.x) < abs(ring_mcp.x - wrist.x) * 1.2
    pinky_curled = abs(pinky_tip.x - wrist.x) < abs(pinky_mcp.x - wrist.x) * 1.2
    
    # Index should be more extended than other fingers
    index_most_extended = (abs(index_tip.x - wrist.x) > abs(middle_tip.x - wrist.x) and
                          abs(index_tip.x - wrist.x) > abs(ring_tip.x - wrist.x))
    
    return (index_extended and horizontal_pointing and 
            middle_curled and ring_curled and pinky_curled and 
            index_most_extended)


def detect_fist_raised(hand_lm):
    """
    Detects closed fist with arm raised (either horizontally or vertically)
    All fingers curled, arm elevated
    """
    wrist = hand_lm.landmark[0]
    
    # Check all finger tips
    index_tip = hand_lm.landmark[8]
    middle_tip = hand_lm.landmark[12]
    ring_tip = hand_lm.landmark[16]
    pinky_tip = hand_lm.landmark[20]
    thumb_tip = hand_lm.landmark[4]
    
    # Check all finger MCPs (bases)
    index_mcp = hand_lm.landmark[5]
    middle_mcp = hand_lm.landmark[9]
    ring_mcp = hand_lm.landmark[13]
    pinky_mcp = hand_lm.landmark[17]
    
    # All fingers should be curled (tips closer to wrist than bases)
    index_curled = abs(index_tip.y - wrist.y) < abs(index_mcp.y - wrist.y) + 0.02
    middle_curled = abs(middle_tip.y - wrist.y) < abs(middle_mcp.y - wrist.y) + 0.02
    ring_curled = abs(ring_tip.y - wrist.y) < abs(ring_mcp.y - wrist.y) + 0.02
    pinky_curled = abs(pinky_tip.y - wrist.y) < abs(pinky_mcp.y - wrist.y) + 0.02
    
    all_curled = index_curled and middle_curled and ring_curled and pinky_curled
    
    # Arm should be raised (wrist Y position should be high - lower value in screen coordinates)
    arm_raised = wrist.y < 0.6  # Upper 60% of screen
    
    return all_curled and arm_raised


# ==========================================
#   PLAYBACK ENGINE
# ==========================================

def activate_gif(gif_frames):
    global active_mode, current_gif_frames, current_gif_index
    active_mode = "gif"
    current_gif_frames = gif_frames
    current_gif_index = 0
    video_done_flag.set()
    play_video_flag.clear()


def activate_image(img):
    global active_mode, current_gif_frames
    active_mode = "image"
    current_gif_frames = [img]
    video_done_flag.set()
    play_video_flag.clear()


def activate_video(path):
    global active_mode, CURRENT_VIDEO
    active_mode = "video"
    CURRENT_VIDEO = path
    video_done_flag.clear()
    play_video_flag.set()


def set_idle_state():
    global active_mode, current_gif_frames, current_gif_index
    active_mode = "idle"
    current_gif_frames = []
    current_gif_index = 0
    play_video_flag.clear()
    video_done_flag.set()


# ==========================================
#         PRIORITY ORDER FOR GESTURES
# ==========================================

def evaluate_gesture_priority(
        shy_gesture,
        angry_face, eyebrows_raised, biting_teeth,
        both_hands_chin, hand_on_head, pointing_at_camera, fist_raised
    ):
    if fist_raised:
        return "fist_raised"
    if pointing_at_camera:
        return "pointing_at_camera"
    if shy_gesture:
        return "shy_gesture"
    if eyebrows_raised:
        return "eyebrows_raised"
    if hand_on_head:
        return "hand_on_head"
    if both_hands_chin:
        return "both_hands_chin"
    if angry_face:
        return "angry_face"
    if biting_teeth:
        return "biting_teeth"
    return "none"


# ==========================================
#                 MAIN LOOP
# ==========================================

def run_opencv_app():
    """Main function to run the OpenCV application"""
    global cooldown, idle_hold
    
    cap = cv2.VideoCapture(0)
    frame_counter = 0
    gesture_hold = 0
    cooldown = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cam_h = 480
        cam_w = 640

        resized_cam = cv2.resize(frame, (cam_w, cam_h))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_result = face_mesh.process(rgb)
        hands_result = hands.process(rgb)

        face_lm = None
        hand_lms = []

        if face_result.multi_face_landmarks:
            face_lm = face_result.multi_face_landmarks[0].landmark
            nose_x_buffer.append(face_lm[1].x)
            nose_y_buffer.append(face_lm[1].y)

        if hands_result.multi_hand_landmarks:
            hand_lms = hands_result.multi_hand_landmarks

        # Detect all gestures
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

        highest = evaluate_gesture_priority(
            shy_gesture,
            angry_face, eyebrows_raised, biting_teeth,
            both_hands_chin, hand_on_head, pointing_at_camera, fist_raised
        )

        if cooldown > 0:
            cooldown -= 1
            highest = "none"

        if highest != "none":
            gesture_hold += 1
        else:
            gesture_hold = 0

        if gesture_hold >= REQUIRED_FRAMES and cooldown == 0:
            print("🎯 Trigger:", highest)
            gesture_hold = 0
            cooldown = COOLDOWN_FRAMES

            if highest == "fist_raised":
                activate_gif(GIF_ARM_RAISE)
            elif highest == "pointing_at_camera":
                activate_video(PATH_INDEX_POINTING)
            elif highest == "shy_gesture":
                activate_gif(GIF_SHY_BITE)
            elif highest == "hand_on_head":
                activate_gif(GIF_HAND_ON_HEAD)
            elif highest == "angry_face":
                activate_gif(GIF_ANGRY_FACE)
            elif highest == "eyebrows_raised":
                activate_gif(GIF_EYEBROWS_UP)
            elif highest == "biting_teeth":
                activate_gif(GIF_BITING_TEETH)
            elif highest == "both_hands_chin":
                activate_gif(GIF_BOTH_HANDS_CHIN)

        # BUILD RIGHT PANEL
        if active_mode == "gif":
            if len(current_gif_frames) > 0:
                frame_to_show = current_gif_frames[current_gif_index]
                current_gif_index += 1
                if current_gif_index >= len(current_gif_frames):
                    set_idle_state()
                frame_to_show = cv2.resize(frame_to_show, (cam_w, cam_h))
            else:
                frame_to_show = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

        elif active_mode == "image":
            if len(current_gif_frames) > 0:
                frame_to_show = cv2.resize(current_gif_frames[0], (cam_w, cam_h))
            else:
                frame_to_show = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

        elif active_mode == "video":
            if not video_queue.empty():
                frame_to_show = cv2.resize(video_queue.get(), (cam_w, cam_h))
            elif video_done_flag.is_set():
                set_idle_state()
                frame_to_show = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            else:
                frame_to_show = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

        elif active_mode == "idle":
            frame_to_show = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

        else:
            frame_to_show = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

        # SAFETY NORMALIZATION
        if frame_to_show is None or not isinstance(frame_to_show, np.ndarray):
            frame_to_show = np.zeros_like(resized_cam)

        if len(frame_to_show.shape) == 2:
            frame_to_show = cv2.cvtColor(frame_to_show, cv2.COLOR_GRAY2BGR)

        if frame_to_show.shape[2] == 4:
            frame_to_show = frame_to_show[:, :, :3]

        frame_to_show = cv2.resize(frame_to_show, (resized_cam.shape[1], resized_cam.shape[0]))

        combined = cv2.hconcat([resized_cam, frame_to_show])
        cv2.imshow("Freak Detector", combined)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_opencv_app()
