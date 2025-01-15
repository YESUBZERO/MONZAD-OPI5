import cv2
from config.config import VIDEO_PATH

def load_video(video_path = None):
    video_path = video_path or VIDEO_PATH
    if not video_path:
        raise ValueError("VIDEO_PATH no está definido.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"No se puede abrir el video en la ruta: {video_path}")
    return cap

def process_video(video_path=None):
    cap = load_video(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    # Liberar recursos
    cap.release()