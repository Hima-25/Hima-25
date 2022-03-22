import cv2
import os


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


face_id = input('enter your id ')
cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier("D:\pycharm\Attendance\haarcascade_frontalface_default.xml")
count = 0

assure_path_exists("D:\pycharm\Attendance\\facedetection")
while (True):
    _, image_frame = cap.read()
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("D:\pycharm\Attendance\\facedetection" + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
        cv2.imshow('frame', image_frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    elif count >= 50:
        print("Successfully Captured")
        break

cap.release()
cv2.destroyAllWindows()