import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import resize
import os

# Load the bacteria  detection model
bacteria = torch.hub.load('ultralytics/yolov5', 'custom', path='bacteria/weights/best.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bacteria = bacteria.to(device)
bacteria.eval()

# Define the list of classes for detection
classes = ['E.coli']

# Specify the image directory
image_directory = 'E.coli-Detection-2/test/images'
image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    frame = cv2.imread(image_path)

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
    boxes_bacteria = results_bacteria.xyxy[0].cpu().numpy()
    labels_bacteria = results_bacteria.xyxyn[0].cpu().numpy()[:, -1].astype(int)

    # Annotate and display the frame
    for box, label in zip(boxes_bacteria, labels_bacteria):
        cls = int(label)
        conf = box[4]
        confidence_threshold = 0.5
        if cls < len(classes) and classes[cls] == 'E.coli' and conf > confidence_threshold:
            xmin, ymin, xmax, ymax = map(int, box[:4])
            color = (0, 255, 0)
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            circle_center_x = (xmin + xmax) // 2
            circle_center_y = (ymin + ymax) // 2
            circle_center = (circle_center_x, circle_center_y)
            circle_radius = 2
            frame = cv2.circle(frame, circle_center, circle_radius, color, -1)
            frame = cv2.putText(frame, f"E.coli {conf:.2f}", (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Display the annotated frame
        frame_display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Bacteria Detection', frame_display)

        # Check for user input to navigate
        key = cv2.waitKey(0) & 0xFF

        if key == ord('n'):
            continue  # Proceed to the next image
        elif key == ord('q'):
            break

# Close windows
cv2.destroyAllWindows()
