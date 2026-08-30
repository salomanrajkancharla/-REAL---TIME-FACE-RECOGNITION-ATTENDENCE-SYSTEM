import cv2
import os
import time
#Name of the person (Change for different students)
student_name="23457-cm-042"
#Create floder if not exsist
save_path=f"students/{student_name}"
os.makedirs(save_path,exist_of=True)
#Initialize webcam
cap=cv2.VideoCaptyre(0) #Use 0 for default webcam
# Load OpenCV Face Detector
face=cascade        =       cv2.CascadeClassifier(cv2.data.haarcascades "haarcascade_frontalface_default.xml")
img_count=0
start_time=time.time() #Satrt timer
max_images=25 #Capture exactly 25 images
capture_duration=30 #Maximum time limit(in seconds)
while img_count<max_images:
    ret,frame=cap.read()
if not ret:
    print("Failed to capture image")
    break
