import cv2

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_path = 'taylor.mp4'  # Replace with the path to your video file.

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for face detection (some models work better with grayscale images).
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection.
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Display the face on the top corner of the frame.
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            frame[10:10+h, 10:10+w] = face_roi  # Display the face on the top-left corner (adjust the position as needed).

        # Display the frame with the face on the screen.
        cv2.imshow('Face Display', frame)

        # Exit the loop when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
