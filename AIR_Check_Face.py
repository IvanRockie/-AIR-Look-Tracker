import cv2
import dlib

# Инициализация распознавания лица
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Загрузите модель

# Загрузка видео
video_path = "path_to_video.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = landmarks.part(36).x, landmarks.part(36).y
        right_eye = landmarks.part(45).x, landmarks.part(45).y

        # Рисование точек на глазах
        cv2.circle(frame, left_eye, 2, (0, 255, 0), -1)
        cv2.circle(frame, right_eye, 2, (0, 255, 0), -1)

    cv2.imshow("Video Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
