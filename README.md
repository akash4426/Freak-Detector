# 🎭 Freak Detector — Real-Time Gesture-Controlled Meme Engine

<div align="center">

**Transform your reactions into instant memes with AI-powered gesture recognition**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

</div>

---

## 🚀 What is Freak Detector?

**Freak Detector** is an advanced real-time computer vision system that captures your facial expressions and hand gestures through your webcam and **instantly responds with the perfect meme**. 

Built with cutting-edge **MediaPipe AI**, **OpenCV**, and **Python**, this project creates an interactive meme playground where your emotions control the content.

### ✨ Key Features

- 🎯 **8+ Gesture Recognition Patterns** — Facial expressions and hand gestures detected in real-time
- ⚡ **Instant Response** — Sub-second meme playback with zero lag
- 🎬 **Multi-Format Support** — Plays GIFs, videos, and images dynamically
- 🧠 **Smart Priority System** — Ensures the right meme plays for combined gestures
- 🔄 **Cooldown Management** — Prevents gesture spam for smooth experience
- 📹 **Dual Display** — Live webcam + meme side-by-side visualization

---

Features

**Face Gestures**

| Gesture                    | Description                                   | Meme              |
| -------------------------- | --------------------------------------------- | ----------------- |
| 😛 Tongue out + head shake | Shake head sideways while sticking tongue out | `freaky-orca.gif` |
| 🟢 Head nod                | Quick up-down movement                        | `ishowspeed.gif`  |
| 😐 Idle stare              | Looking straight at the camera, no movement   | `monkeytruth.jpg` |



**Hand Gestures**

| Gesture                   | Description                        | Meme                 |
| ------------------------- | ---------------------------------- | -------------------- |
| 🤲 Rubbing palms together | Hands close, moving back and forth | `freaky-sonic.mp4`   |
| 😩 Both hands on head     | “Oh no” reaction                   | `ishowspeed-wow.gif` |
| ☝️ One finger up          | Index finger raised                | `monkeyrealize.jpeg` |
| 🤔 Hand on chin           | Thinking pose                      | `monkeythink.jpg`    |
| 👍 Thumbs up              | Positive gesture                   | `thumbsupmonkey.png` |


How It Works

### 🧠 **AI-Powered Recognition**

#### **MediaPipe FaceMesh**
- Tracks **468 facial landmarks** in real-time
- Analyzes eyebrow positions, eye openness, mouth shape
- Detects micro-expressions with high accuracy
- Processes at 30+ FPS for smooth detection

#### **MediaPipe Hands**
- Tracks **21 landmarks per hand** (up to 2 hands simultaneously)
- 3D hand pose estimation in real-time
- Finger position tracking for precise gesture detection
- Palm orientation and hand shape analysis

#### **OpenCV Processing**
- High-speed webcam capture and frame processing
- Dual-panel rendering (webcam + meme output)
- Multi-format media playback (GIF/MP4/Image)
- Optimized frame buffering for smooth playback

### ⚙️ **Smart Gesture Engine**

- **Sustained Detection**: Requires consistent gesture hold (6 frames) to prevent false triggers
- **Cooldown System**: 30-frame pause between gestures for clean transitions
- **Priority Hierarchy**: Resolves conflicting gestures intelligently
- **Threaded Video Playback**: Non-blocking video rendering for performance
- **Frame Buffering**: Smooth GIF playback with pre-loaded frames



## 📺 Output Display

The application features a **dual-panel interface** for immersive interaction:

```
╔═══════════════════════════╦═══════════════════════════╗
║       LIVE WEBCAM         ║     MEME RESPONSE         ║
║                           ║                           ║
║   👤 Real-time tracking   ║   🎬 Dynamic playback     ║
║   🔴 Face landmarks       ║   ⚡ Instant updates      ║
║   ✋ Hand detection       ║   🎭 Context-aware memes  ║
╚═══════════════════════════╩═══════════════════════════╝
```

**Left Panel**: Your live webcam feed with gesture tracking
**Right Panel**: Responsive meme playback triggered by your gestures

✨ Gestures trigger memes **instantly** with zero noticeable delay!


## 📁 Project Structure

```
📦 freak-detector/
├── 🐍 freakdetector.py          # Main application logic
├── 📋 requirements.txt          # Python dependencies
├── 📖 README.md                 # Documentation
└── 🎬 memes/                    # Meme asset library
    ├── pottodu-rakhi.gif        # Shy gesture response
    ├── pottodu-angry.gif        # Angry face response
    ├── pottodu-cries.gif        # Both hands chin response
    ├── pottodu-liftingeyebrows.gif  # Eyebrows raised response
    ├── pottodu-bitingteeth.gif  # Biting teeth response
    ├── ammathodu-pottodu.gif    # Hand on head response
    ├── thaali-pottodu-converted.mp4  # Pointing gesture response
    └── sai.gif                  # Fist raised response
```

---

## 🛠️ Installation

### Prerequisites
- **Python 3.10+** installed
- Working **webcam** (720p or higher recommended)
- 4GB+ RAM for smooth performance

### Quick Start

**1️⃣ Clone the Repository**

```bash
git clone https://github.com/akash4426/Freak-Detector.git
cd Freak-Detector
```

**2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install opencv-python mediapipe numpy pillow
```

**3️⃣ Launch the Application**

```bash
python freakdetector.py
```

🎥 **Make sure your webcam is enabled and grant camera permissions when prompted**

---

## 🎮 Usage

1. **Launch** the application
2. **Position** yourself in front of the webcam
3. **Perform** any gesture from the supported list
4. **Watch** as the perfect meme appears instantly!

💡 **Pro Tip**: Hold gestures for ~200ms to ensure detection. Quick flashes won't trigger memes.

**Exit**: Press `ESC` key to close the application



## 🎨 Customization

Freak Detector is highly customizable! Here's what you can modify:

### 🎭 **Add Custom Gestures**
Create new detection functions in the gesture detection section:
```python
def detect_your_gesture(face_lm, hand_lm):
    # Your custom logic here
    return True/False
```

### 🖼️ **Replace Memes**
Drop your own GIFs/videos/images into `/memes/` and update file paths:
```python
PATH_YOUR_MEME = "memes/your-meme.gif"
```

### ⚙️ **Tune Sensitivity**
```python
REQUIRED_FRAMES = 6      # Lower = more sensitive
COOLDOWN_FRAMES = 30     # Adjust pause between gestures
```

### 🎯 **Change Priority**
Modify `evaluate_gesture_priority()` to control which gesture wins when multiple are detected.

### 🔊 **Add Audio**
Integrate `pygame` or `playsound` to play sound effects alongside memes!

## 📋 Requirements

| Component | Version | Purpose |
|-----------|---------|----------|
| **Python** | 3.10+ | Core runtime |
| **OpenCV** | 4.8+ | Computer vision & rendering |
| **MediaPipe** | 0.10+ | AI gesture/face detection |
| **NumPy** | 1.24+ | Array operations |
| **Pillow** | 10.0+ | GIF frame loading |
| **Webcam** | 720p+ | Video input |

### System Requirements
- **OS**: Windows 10/11, macOS 10.15+, Linux (Ubuntu 20.04+)
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: Modern multi-core processor for real-time processing


## 🤝 Contributing

**Contributions are welcome!** Help make Freak Detector even better:

### 🌟 Ideas for Contributions
- 🎯 **New Gesture Types**: Add more facial expressions or hand signs
- 🎨 **Meme Packs**: Curated collections for different themes
- 🖥️ **GUI Interface**: Settings panel for real-time adjustments
- 👥 **Multi-Person Support**: Detect multiple users simultaneously
- 🔊 **Audio Integration**: Add sound effects and voice reactions
- 🌍 **Localization**: Multi-language meme support
- 📱 **Mobile Port**: Android/iOS companion app

### 📝 How to Contribute
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### 🐛 Bug Reports
Found a bug? [Open an issue](https://github.com/akash4426/Freak-Detector/issues) with:
- Clear description
- Steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🎉 Have Fun Freaking Out!

<div align="center">

### This project reacts to your emotions in real-time — let the memes fly! 🚀

**Made with ❤️ by [Akash](https://github.com/akash4426)**

⭐ **Star this repo** if you found it useful!

🐛 **Report issues** | 💡 **Suggest features** | 🤝 **Contribute**

---

### 📱 Connect

[![GitHub](https://img.shields.io/badge/GitHub-akash4426-black?style=for-the-badge&logo=github)](https://github.com/akash4426)

---

*Real-time gesture recognition meets meme culture* 🎭

</div>
