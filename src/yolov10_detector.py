from rknnlite.api import RKNNLite as RKNN
from config.config import MODEL_PATH
import cv2, numpy as np

def load_model(model_path = None):
    model_path = model_path or MODEL_PATH
    rknn = RKNN()
    model = rknn.load_rknn(model_path)
    if model != 0:
        raise RuntimeError(f"Fallo al cargar el modelo RKNN en la ruta {model_path}")
    
    print('--> Init Runtime environment')
    model = rknn.init_runtime(core_mask=rknn.NPU_CORE_ALL)
    if model != 0:
        print('Init runtime environment failed')
        exit(model)
    print('done')

    return rknn

def detect_objects(rknn, frame):
    input_frame_ori = cv2.resize(frame, (640, 640))  # Ajustar tamaño según tu modelo
    input_frame = np.expand_dims(input_frame_ori, axis=0)
    outputs = rknn.inference(inputs=[input_frame])

    return outputs