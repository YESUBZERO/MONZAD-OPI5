import os
from dotenv import load_dotenv

load_dotenv()

VIDEO_PATH = os.getenv("VIDEO_PATH")
MODEL_PATH = os.getenv("MODEL_PATH")