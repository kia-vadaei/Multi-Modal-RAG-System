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

class Chunking():
    def get_avg_frame_per_time(self):
        try: 
            avg_frame_per_chunk = sum([len(self.chunks[i]) for i in range(len(self.chunks))]) / len(self.chunks)
            return avg_frame_per_chunk
        except Exception as e:
            print(f'An error while running avg_frame_per_time: {e}')

    def get_mean_entropy(self):
        mean_entropy = E.chunks_mean_entropy(self.chunks)
        return mean_entropy
    
    def add_to_table(self, table, approch):
        dict_temp = {}
        dict_temp['Approach'] = approch
        dict_temp['Execution Time'] = self.exe_time
        dict_temp['Avg Frame/Chunk'] = self.get_avg_frame_per_time()
        dict_temp['Mean Entropy'] = self.get_mean_entropy()
        dict_temp['Num Chunks'] = len(self.chunks)
        table.append(dict_temp.copy())
        return table

class ClipChunking(Chunking):
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
    def evaluate(self):
        self.avg_frame_per_chunk = super().get_avg_frame_per_time()
        self.mean_entropy = super().get_mean_entropy()


class SaliencyChunking(Chunking):
    def __init__(self):
        self.chunks = None
        self.exe_time = None

    def spectral_residual(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f_transform = np.fft.fft2(gray_frame)
        f_transform_shifted = np.fft.fftshift(f_transform)

        magnitude = np.abs(f_transform_shifted)
        log_magnitude = np.log1p(magnitude)  # log(1 + x) to avoid log(0)
        spectral_residual = np.exp(log_magnitude - np.mean(log_magnitude))

        return spectral_residual
    
    def detect_slide_changes(self, video_path, threshold=0.125, interval=1):
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()

        frame_lst = []
        Saliency_chunks = []

        if not success:
            print("Error: Failed to read the first frame.")
            return []

        prev_spectral_residual = self.spectral_residual(frame)
        timestamp = 0
        timestamps = []

        while cap.isOpened():
            # Move to the next frame based on the interval
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            if not ret:
                break

            spectral_residual_frame = self.spectral_residual(frame)
            mean_residual = np.mean(spectral_residual_frame)

            if abs(mean_residual - np.mean(prev_spectral_residual)) > threshold:
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                print(f"Slide changed at {minutes:02}:{seconds:02} minutes")  # Print slide change time
                timestamps.append(timestamp)
                Saliency_chunks.append(frame_lst)
                frame_lst = []

            prev_spectral_residual = spectral_residual_frame

            frame_lst.append(frame)

            timestamp += interval

        cap.release()
        return Saliency_chunks, timestamps

    
    def chunk(self, video_path):
        start_time = time()
        self.chunks, slide_change_timestamps = self.detect_slide_changes(video_path, interval=1)
        end_time = time()
        self.exe_time = end_time - start_time
        return self.chunks, slide_change_timestamps, self.exe_time
    
class SSIMChunking(Chunking):
    def __init__(self):
        self.chunks = None
        self.exe_time = None

    def detect_slide_changes(video_path, threshold=0.8, interval=1):
        cap = cv2.VideoCapture(video_path)
        frame_lst = []
        ssim_chunks = []
        # Attempt to read the first frame
        success, prev_frame = cap.read()
        if not success:
            print("Error: Failed to read the first frame.")
            return []

        # Convert the first frame to grayscale
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        timestamp = 0

        # Store timestamps of slide changes
        timestamps = []

        while cap.isOpened():
            # Set the capture to the next frame based on interval
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert current frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate SSIM between previous and current frame
            similarity = ssim(prev_frame, gray_frame)

            # If similarity is below the threshold, it's a slide change
            if similarity < threshold:
                # Store the timestamp
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                print(f"Slide changed at {minutes:02}:{seconds:02} minutes")  # Print slide change time
                timestamps.append(timestamp)
                ssim_chunks.append(frame_lst)
                frame_lst = []

            # Update the previous frame
            prev_frame = gray_frame
            frame_lst.append(frame)
            timestamp += interval

        cap.release()
        return ssim_chunks, timestamps


    def chunk(self, video_path):
        start_time = time()
        self.chunks, slide_change_timestamps = self.detect_slide_changes(video_path, interval=1)
        end_time = time()
        self.exe_time = end_time - start_time
        return self.chunks, slide_change_timestamps, self.exe_time