import os
import random
import time
import cv2
from ultralytics import YOLO

#Connect GND to physical pin 6)

# LED setup (WS2812 via GPIO21)
import board
import neopixel

# Servo setup
import RPi.GPIO as GPIO

# Raspberry Pi Camera
from picamera2 import Picamera2

# ---------------------------
# NeoPixel LED Initialization (connect to physical pin 12)
# ---------------------------
NUM_PIXELS = 8  # Number of LEDs in the strip
pixels = neopixel.NeoPixel(board.D21, NUM_PIXELS, brightness=0.5, auto_write=True)

# Function to set LED color
def set_led(r, g, b):
    pixels.fill((r, g, b))

# ---------------------------
# Servo Initialization
# ---------------------------
SERVO_PIN = 12  # GPIO12 (Pin 32) connected to jaw servo
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM frequency
pwm.start(0)

# Function to set the angle of the jaw
def set_jaw(angle):
    duty = 2 + (angle / 18)  # Convert angle to PWM duty cycle
    GPIO.output(SERVO_PIN, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)
    GPIO.output(SERVO_PIN, False)
    pwm.ChangeDutyCycle(0)

# Convenience functions for opening and closing jaw
def open_jaw():
    set_jaw(0)

def close_jaw():
    set_jaw(150)

# ---------------------------
# YOLO Model Initialization
# ---------------------------
model = YOLO('my_model.pt')  # Load your custom trained model
labels = model.names         # Class labels (e.g., circle, square, etc.)

# ---------------------------
# Camera Initialization
# ---------------------------
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))  # Set resolution
print("cam initialized")
picam2.start()
print("cam started")
time.sleep(2)
print("slept 2")

# ---------------------------
# Main Game Loop
# ---------------------------
shapes = ["circle", "square", "heart", "triangle"]  # List of possible target shapes
MAX_OPEN_TIME = 5  # Max time to keep jaw open

try:
    while True:
        # Select a random target shape
        target_shape = random.choice(shapes)

        # Play start and shape name audio
        os.system("aplay start.wav")
        os.system(f"aplay {target_shape}.wav")
        print("espeakdone")

        # Close jaw and light up white
        close_jaw()
        set_led(255, 255, 255)  # White light indicates ready state
        time.sleep(5)  # Give user time to present object

        # Capture a frame from the camera
        frame = picam2.capture_array()

        # Convert frame to RGB if needed
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO detection
        results = model(frame, verbose=False)
        detections = results[0].boxes

        found_shape = None
        print("Detections:")
        for det in detections:
            conf = det.conf.item()  # Get confidence score
            if conf < 0.5:
                continue  # Skip low confidence detections

            classid = int(det.cls.item())
            label = labels[classid]  # Get class label
            print(f" - {label} ({conf:.2f})")

            if label.lower() == target_shape.lower():
                found_shape = target_shape
                break  # Stop loop if correct shape is found
            elif not found_shape:
                found_shape = label  # Store first found shape if none match target

        # If the correct shape was found
        if found_shape and found_shape.lower() == target_shape.lower():
            os.system("aplay correct.wav")
            set_led(0, 255, 0)  # Green LED for correct
            time.sleep(1)
            open_jaw()
        else:
            os.system("aplay wrong.wav")
            set_led(255, 0, 0)  # Red LED for wrong

        # Keep jaw open for a bit before resetting
        jaw_open_time = time.time()
        while True:
            if cv2.waitKey(10) == 27:  # Exit on ESC key (if window is used)
                break
            if found_shape == target_shape:
                break
            if time.time() - jaw_open_time > MAX_OPEN_TIME:
                close_jaw()
                break

        # Turn off LED and pause before next round
        set_led(0, 0, 0)
        time.sleep(1)

# Shutdown using Ctrl+C
except KeyboardInterrupt:
    print("Exiting...")

# Cleanup GPIO and camera resources
finally:
    set_led(0, 0, 0)
    close_jaw()
    pwm.stop()
    GPIO.cleanup()
    picam2.stop()
    cv2.destroyAllWindows()
