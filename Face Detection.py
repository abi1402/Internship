import cv2

alg = "haarcascade_frontalface_default.xml"

haar_cascade = cv2.CascadeClassifier(alg)

cam = cv2.VideoCapture(0)

total_faces = 0
correct_faces = 0

while True:
    _, img = cam.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4)
    
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        total_faces += 1

    # Increment correct_faces if the face is detected correctly
    if len(face) > 0:
        correct_faces += 1
    
    cv2.imshow("FaceDetection", img)
    key = cv2.waitKey(10)
    if key == 27:
        break

# Calculate the accuracy after the face detection loop completes
if total_faces > 0:
    accuracy = (correct_faces / total_faces) * 100
    print(f"Total Accuracy: {accuracy:.2f}%")

cam.release()
cv2.destroyAllWindows()

