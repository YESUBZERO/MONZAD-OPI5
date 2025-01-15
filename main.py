from src.video_processor import process_video
from src.yolov10_detector import load_model, detect_objects
from src import post_process_yolov10
from src import draw
import cv2


CLASSES = ('pollo',)
OBJ_THRESH = 0.60

def main():
    rknn = load_model()
    for frame in process_video():
        detections, frame_ori = detect_objects(rknn, frame)
        boxes, classes, scores = post_process_yolov10.post_process_yolov10(detections, CLASSES, OBJ_THRESH)
        img_p = frame_ori.copy()
        if boxes is not None:
            draw.draw(img_p, boxes, scores, classes, CLASSES)
            cv2.imshow("full post process result", img_p)
            # Salir si se presiona la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    # Desinicializar el runtime
    rknn.release()


if __name__ == "__main__":
    main()