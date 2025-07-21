import cv2
import time
import RPi.GPIO as GPIO
from ultralytics import YOLO

# -------------------- SERVO SETUP --------------------
SERVO_PIN = 12  # Physical pin 32 (GPIO12)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)

def set_angle(angle):
    duty = 2 + (angle / 18)
    GPIO.output(SERVO_PIN, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)
    GPIO.output(SERVO_PIN, False)
    pwm.ChangeDutyCycle(0)

# -------------------- YOLO SETUP --------------------
model = YOLO("my_model.pt")  # Replace with yolov8n.pt if needed
target_class = "person"      # Change to the object you're detecting

# Open default camera (0)
cap = cv2.VideoCapture(0)
cap.set(3, 320)  # width
cap.set(4, 240)  # height

print("[INFO] Starting detection...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error.")
            break

        results = model(frame, stream=True)

        object_found = False
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                if cls_name == target_class:
                    object_found = True
                    break

        # Control servo based on detection
        if object_found:
            print("Detected:", target_class)
            set_angle(0)   # Open
        else:
            print("Not detected")
            set_angle(90)  # Closed

        # Optional: show frame (slow on headless Pi)
        # cv2.imshow("YOLOv8", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

except KeyboardInterrupt:
    print("Exiting...")

finally:
    cap.release()
    pwm.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
