import cv2
import face_recognition
import numpy as np

# Load the known face images
known_faces = [face_recognition.load_image_file("D:/AI/face/Photo2.jpg"), face_recognition.load_image_file("D:/AI/face/Photo3.jpg")]
known_face_encodings = [face_recognition.face_encodings(face)[0] for face in known_faces]

known_face2 = face_recognition.load_image_file("D:/AI/face/Photo.jpg")
known_face_encoding2 = face_recognition.face_encodings(known_face2)[0]

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    #rgb_frame = frame[:, :, ::-1]
    rgb_frame = cv2.resize(frame, (640, 480))

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)

            if True in matches:
                label = "Verified Successful"
                color = (0, 255, 0)  # Green
                # Draw a box around
                cv2.rectangle(rgb_frame, (left, top), (right, bottom), color, 2)
            else:
                label = "Verified Failed"
                color = (0, 0, 255)  # Red

            # label text show
            cv2.putText(rgb_frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    else:
        print("未检测到面部")

    cv2.imshow("Face Detection", rgb_frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()