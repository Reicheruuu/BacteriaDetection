import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import resize

classes = ['E.coli']

bacteria = torch.hub.load('ultralytics/yolov5', 'custom', path='bacteria/weights/best.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_person = bacteria.to(device)
bacteria.eval()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape
    max_size = 900
    if width > height:
        new_width = max_size
        new_height = int(height / (width / max_size))
    else:
        new_height = max_size
        new_width = int(width / (height / max_size))
    frame = resize(Image.fromarray(frame), (new_height, new_width))

    # Convert frame to numpy array
    frame = np.array(frame)

    # Perform inference with bacteria detection model

    results_bacteria = bacteria(frame)

    # Get detected objects and their position for bacteria detection

    boxes_bacteria = results_bacteria.xyxy[0].cpu().numpy()
    labels_bacteria = results_bacteria.xyxyn[0].cpu().numpy()[:, -1].astype(int)

    for box, label in zip(boxes_bacteria, labels_bacteria):
        cls = int(label)
        conf = box[4]
        confidence_threshold = 0.5
        if cls < len(classes) and classes[cls] == 'E.coli' and conf > confidence_threshold:
            xmin, ymin, xmax, ymax = map(int, box[:4])
            color = (255, 0, 0)
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            circle_center_x = (xmin + xmax) // 2
            circle_center_y = (ymin + ymax) // 2
            circle_center = (circle_center_x, circle_center_y)
            circle_radius = 2

            frame = cv2.circle(frame, circle_center, circle_radius, color, -1)

        # Display the annotated frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Bacteria Detection', frame)

    # Check for user input to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()