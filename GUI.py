import cv2
import sys
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import resize
import os
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from PyQt5.uic import loadUiType


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
# Load the UI file and extract the Ui_MainWindow class

ui, _ = loadUiType('GUI.ui')
class MainApp(QMainWindow, ui):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)

        # Set the window title
        self.setWindowTitle("Fruit Detection")

        # Set the initial tab to the Tab 0
        self.tabWidget.setCurrentIndex(0)

        # Connect the login button to the login method
        self.MAIN_LOGIN.clicked.connect(self.login)

        # Connect the close button to the close_window method
        self.MAIN_CLOSE.clicked.connect(self.close_window)

        self.NEXT.clicked.connect(self.next)

        self.BACK.clicked.connect(self.back)

        self.START.clicked.connect(self.start)

        self.PREVIOUS.clicked.connect(self.previous)

        # Create an attribute to keep track of the current image index
        self.current_image_index = 0

        # Create a QTimer for image processing
        self.image_processing_timer = QTimer()
        self.image_processing_timer.timeout.connect(self.start_processing)


    ### LOGIN PROCESS ###
    def login(self):
        # Get the entered username and password
        un = self.USERNAME.text()
        un = un.lower()
        pw = self.PASSWORD.text()
        pw = pw.lower()

        # Check if the username and password are correct
        if (un == "erovoutika") and (pw == "123!"):
            # If the username and password are correct, clear the fields and switch to the main tab
            self.USERNAME.setText("")
            self.PASSWORD.setText("")
            self.tabWidget.setCurrentIndex(1)

        else:
            # If the username and/or password are incorrect, show an error message
            if (un != "erovoutika") and (pw == "123!"):
                msg = QMessageBox()
                msg.setText("The username you’ve entered is incorrect.")
                msg.setWindowTitle("INCORRECT USERNAME!")
                msg.setIcon(QMessageBox.Information)
                msg.setStyleSheet("background-color: rgb(36, 13, 74);color: rgb(255, 255, 255);")
                msg.exec_()
            elif (un == "erovoutika") and (pw != "123!"):
                msg = QMessageBox()
                msg.setText("The password you’ve entered is incorrect.")
                msg.setWindowTitle("INCORRECT PASSWORD!")
                msg.setIcon(QMessageBox.Information)
                msg.setStyleSheet("background-color: rgb(36, 13, 74);color: rgb(255, 255, 255);")
                msg.exec_()
            else:
                msg = QMessageBox()
                msg.setText("Please enter the correct username and password.")
                msg.setWindowTitle("USERNAME AND PASSWORD!")
                msg.setIcon(QMessageBox.Information)
                msg.setStyleSheet("background-color: rgb(36, 13, 74);color: rgb(255, 255, 255);")
                msg.exec_()

        # Clear the username and password fields
        self.USERNAME.setText("")
        self.PASSWORD.setText("")

### CLOSE WINDOW PROCESS ###
    def close_window(self):
        # Close the window
        self.close()

    def start(self):
        # Start image processing and switch to Tab 1
        self.tabWidget.setCurrentIndex(1)
        self.image_processing_timer.start(1000)  # Set the timer interval as needed

    def start_processing(self):
        if self.current_image_index < len(image_files):
            try:
                image_file = image_files[self.current_image_index]
                image_path = os.path.join(image_directory, image_file)
                frame = cv2.imread(image_path)

                # Resize and process the frame here as in your original code

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
                # Convert the frame to BGR color space
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Convert the frame to QImage
                qImg = QImage(frame_bgr.data, frame_bgr.shape[1], frame_bgr.shape[0], frame_bgr.strides[0],
                              QImage.Format_RGB888)

                # Create a QPixmap from QImage
                pixmap = QPixmap.fromImage(qImg)

                # Set the pixmap to be scaled to the size of the QLabel widget
                self.display.setScaledContents(True)

                # Set the pixmap as the image displayed in the QLabel widget
                self.display.setPixmap(pixmap)

                # Repaint the QLabel widget
                self.display.repaint()

            except Exception as e:
                # Handle any exceptions that occur during image processing
                print(f"Error processing image: {str(e)}")
        else:
            # Stop the timer when all images have been processed
            self.image_processing_timer.stop()

    def back(self):
        # Switch to the login tab (Tab 0) and stop image processing
        self.tabWidget.setCurrentIndex(0)
        self.image_processing_timer.stop()

    def next(self):
        # Increment the image index to go to the next image
        if self.current_image_index < len(image_files) - 1:
            self.current_image_index += 1
            self.start_processing()

    def previous(self):
        # Decrement the image index to go back to the previous image
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.start_processing()

    def closeEvent(self, event):
        self.cap1.release()
        event.accept()

def main():
    # Create an instance of the QApplication class
    app = QApplication(sys.argv)

    # Create an instance of the MainApp class (the main window of the program)
    window = MainApp()

        # Show the window
    window.show()
        # Start the event loop of the application
    app.exec_()

if __name__ == '__main__':
        # Call the main function if this script is being run as the main program
    main()