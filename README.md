# raspi-ai-dino
An interactive, AI-powered toy built with Raspberry Pi, servo motors, and computer vision. This project features a servo-actuated jaw and color-changing LEDs controlled by a Raspberry Pi. It uses a Raspberry Pi camera and a custom-trained YOLOv8n (easily integrated onto Raspberry Pi) object detection model to recognize shapes inserted into the dino's mouth.

# Features
- Servo-controlled jaw — opens and closes when shapes are placed in its mouth.
- Colored LED feedback (green/red) based on shape detection.
- Camera integration — recognizes shapes and colors using machine learning (OpenCV or TensorFlow).
- Detects and responds to various 2D shapes (e.g., circle, triangle, square)
- Raspberry Pi powered — all logic, detection, and control run locally on a Pi.
- Modular, beginner-friendly hardware and codebaseun learning tool for education.

# Shapes Recognized
- Square
- Circle
- Pentagon
- Triangle
Each shape is associated with a color or action (e.g., green light for correct shape, red light for wrong one).

# Hardware Used
- Raspberry Pi 4
- MG90S Servo for jaw movement
- 5V 2.5A DC power supply
- Raspberry Pi camera
- RGB LED module
- Dino frame
- Jumper wires, resistors, glue, screws

# How It Works
- A webcam captures a live video feed of the dino’s mouth.
- A YOLOv8 model detects specific shapes inserted into the mouth.
- If a correct shape is detected:
-   Servo opens the jaw.
-   Green LED turns on.
-   Servo closes and the shape lands in the "stomach" (tray at the base of the dino frame).
- If incorrect:
-   Red LED turns on.
-   Voice lines give the user targeted feedback.

# Requirements
- Python 3.8+
- Ultralytics YOLOv8n
- OpenCV
- NumPy
- RPi.GPIO or gpiozero
- Raspberry Pi OS (Lite or Full)

# Training the Model
- Capture 100–200 JPGs of each shape (circle, triangle, etc.) in varied lighting/backgrounds.
- Create a venv and use Label Studio:
- Install: pip install label-studio
- Start: label-studio start
- Create a project and label shapes in each image.
- Export in YOLO format (YOLO with images).
- Train using YOLOv8n in Edge Electronic's Google Colab or locally:
-   https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb#scrollTo=gzaJQ2sGEPhP
- Run the Model: python raspi.py --model my_model.pt --source picamera --resolution 640x480

# Jaw Mechanism Setup
| Component                  | Connection                                                                  |
| -------------------------- | --------------------------------------------------------------------------- |
| **Power Supply (5V 2.5A)** | Cut and stripped wires manually                                             |
| Red (Power Supply)         | → Red wire on servo (VCC)                                                   |
| Black (Power Supply)       | → Shared Ground: <br>• Raspberry Pi GND pin <br>• Brown wire on servo (GND) |
| Yellow wire (Servo)        | → Raspberry Pi **GPIO12 (pin 32)** (signal pin)                             |
Important: The black wire from the power supply must be connected to the Pi’s GND and the servo’s GND (brown wire) to establish a common ground. This is critical for signal control to work reliably.

# Mechanical Setup
- The servo horn is attached to the MG90S output shaft and screwed in securely.
- A wooden skewer is glued to the horn.
- The other end of the skewer is attached to the dino’s jaw (e.g., a hinge or flap), turning the rotation into up/down movement.
- The servo moves the skewer in a push-pull motion to open and close the jaw.

# How It Works
- When the YOLO model detects a valid shape, the script:
- Sends a PWM signal to GPIO12
- Rotates the servo to a specified angle (e.g., 30° to open, 90° to close)
- The movement triggers a chomping motion, simulating a bite or response
- Servo returns to closed position after a short delay

# LED Setup

# Voice Line Setup
