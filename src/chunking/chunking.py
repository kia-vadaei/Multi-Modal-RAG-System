from pytubefix import YouTube
from pytubefix.cli import on_progress
import os
import cv2
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel, BlipForConditionalGeneration, BlipProcessor, BlipModel
from scipy.spatial.distance import cosine
import os
from skimage.metrics import structural_similarity as ssim
import ffmpeg
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from time import time
from scipy.stats import entropy
from google.colab.patches import cv2_imshow
import pandas as pd
import Entropy as E


class clip_chunking():
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.chunks = None
        self.exe_time = None

    def get_frame_embedding(self, frame):
        inputs = self.processor(images=frame, return_tensors="pt")
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        return embedding.squeeze()
    
    def detect_slide_changes(self, video_path, threshold=0.8, interval=1):
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()

        frame_lst = []
        clip_chunks = []

        if not success:
            print("Error: Failed to read the first frame.")
            return []

        prev_embedding = self.get_frame_embedding(frame)
        timestamp = 0
        timestamps = []

        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            if not ret:
                break

            curr_embedding = self.get_frame_embedding(frame)

            similarity = 1 - cosine(prev_embedding, curr_embedding)

            if similarity < threshold:
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                print(f"Slide changed at {minutes:02}:{seconds:02} minutes")  # Print slide change time
                timestamps.append(timestamp)

                clip_chunks.append(frame_lst)
                frame_lst = []

            prev_embedding = curr_embedding

            frame_lst.append(frame)

            timestamp += interval

        cap.release()
        return clip_chunks, timestamps
    
    def chunk(self, video_path):
        start_time = time()
        self.chunks, slide_change_timestamps = self.detect_slide_changes(video_path, threshold=0.8, interval=1)
        end_time = time()
        self.exe_time = end_time - start_time
        return self.chunks, slide_change_timestamps, self.exe_time
    
    def get_avg_frame_per_time(self):
        try: 
            clip_avg_frame_per_chunk = sum([len(self.chunks[i]) for i in range(len(self.chunks))]) / len(self.chunks)
            return clip_avg_frame_per_chunk
        except Exception as e:
            print(f'An error while running get_avg_frame_per_time: {e}')

    def get_mean_entropy(self):
        clip_mean_entropy = E.chunks_mean_entropy(self.chunks)
        return clip_mean_entropy
    
    def add_to_table(self, table):
        dict_temp = {}
        dict_temp['Approach'] = 'openai/clip-vit-base-patch32 Embbeding Base'
        dict_temp['Execution Time'] = self.exe_time
        dict_temp['Avg Frame/Chunk'] = self.get_avg_frame_per_time()
        dict_temp['Mean Entropy'] = self.get_mean_entropy()
        dict_temp['Num Chunks'] = len(self.clip_chunks)
        table.append(dict_temp.copy())
        return table