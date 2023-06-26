import keras

# Load the model
model = keras.models.load_model('DV.h5')

import cv2
import numpy as np

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class_names = open("labels.txt", "r").readlines()

# Open a video capture object
cap = cv2.VideoCapture(1)

while True:
    # Read frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Preprocess the face region
        face_img = gray[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (160, 160))
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)
        face_img = np.repeat(face_img, 3, axis=-1)
        face_img = face_img / 255.0

        # Perform face recognition
        predictions = model.predict(face_img)
        index = np.argmax(predictions)
        class_name = class_names[index].strip()
        confidence_score = predictions[0][index]

        # Print prediction and confidence score
        print("Class:", class_name)
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if np.round(confidence_score * 100) > 96:
            label = "unknown"
        label = f"{class_name}: {np.round(confidence_score * 100, 2)}%"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
