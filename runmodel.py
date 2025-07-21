import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model (e.g., best.pt)')
parser.add_argument('--source', required=True, help='Image, folder, video file, or "picamera"')
parser.add_argument('--thresh', default=0.5, type=float, help='Confidence threshold')
parser.add_argument('--resolution', default="640x480", help='Resolution WxH')
parser.add_argument('--record', action='store_true', help='Record video output')
args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

if not os.path.exists(model_path):
    print('ERROR: Model path is invalid.')
    sys.exit(1)

resW, resH = map(int, user_res.lower().split('x'))

model = YOLO(model_path)
labels = model.names

source_type = None
if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        source_type = 'image'
    elif ext.lower() in ['.avi', '.mov', '.mp4', '.mkv', '.wmv']:
        source_type = 'video'
    else:
        print(f'Unsupported file extension: {ext}')
        sys.exit(1)
elif img_source == 'picamera':
    source_type = 'picamera'
else:
    print(f'Unknown source: {img_source}')
    sys.exit(1)

if record and source_type not in ['video', 'picamera']:
    print('Recording only supported for video or picamera.')
    sys.exit(1)

if record:
    recorder = cv2.VideoWriter('demo1.avi',
                               cv2.VideoWriter_fourcc(*'MJPG'),
                               30, (resW, resH))

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = glob.glob(os.path.join(img_source, '*'))
    imgs_list = [f for f in imgs_list if os.path.splitext(f)[1].lower() in ['.jpg','.jpeg','.png','.bmp']]
elif source_type == 'video':
    cap = cv2.VideoCapture(img_source)
    cap.set(3, resW)
    cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (resW, resH), "format": "RGB888"}))
    picam2.start()
else:
    print('Unsupported source.')
    sys.exit(1)

bbox_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
               (255, 255, 0), (255, 0, 255), (0, 255, 255)]

frame_rate_buffer = []
fps_avg_len = 30
img_count = 0

while True:
    t_start = time.perf_counter()

    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images processed.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Video ended.')
            break
    elif source_type == 'picamera':
        frame = picam2.capture_array()
        if frame is None:
            print('Failed to capture from camera.')
            break
    else:
        print('Invalid source.')
        break

    frame = cv2.resize(frame, (resW, resH))
    results = model(frame, verbose=False)
    detections = results[0].boxes

    object_count = 0
    for det in detections:
        conf = det.conf.item()
        if conf < min_thresh:
            continue
        classid = int(det.cls.item())
        color = bbox_colors[classid % len(bbox_colors)]
        xmin, ymin, xmax, ymax = map(int, det.xyxy[0].tolist())
        label = f"{labels[classid]} {conf:.2f}"
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        object_count += 1

    t_end = time.perf_counter()
    fps = 1.0 / (t_end - t_start)
    frame_rate_buffer.append(fps)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)

    avg_fps = np.mean(frame_rate_buffer)
    cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Objects: {object_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("YOLO Results", frame)

    if record:
        recorder.write(frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.imwrite('capture.png', frame)
    elif key == ord('s'):
        print('Paused. Press any key to resume...')
        cv2.waitKey(0)

if source_type == 'video':
    cap.release()
elif source_type == 'picamera':
    picam2.stop()

if record:
    recorder.release()

cv2.destroyAllWindows()
