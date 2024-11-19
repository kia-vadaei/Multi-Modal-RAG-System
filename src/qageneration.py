import cv2
import pytesseract
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from pytubefix.cli import on_progress
from pytubefix import YouTube
import numpy as np
import cv2
from IPython.display import display
import os
import json
from langchain_openai import ChatOpenAI  # pip install -U langchain_openai
import base64
import requests
from tqdm import tqdm
class QAGeneration():
    def __init__(self, video_urls = None , chunks_root_path = '../chunks/hybrid_clip_ssim_chunking'):

        base_url = "https://api.avalai.ir/v1"
        api_key = "aa-pVNCqkgpuqSpvC8EdFmr8tzJy9HhzEkz5OGfT8YWxw9pB7kK"
        model_name = "gpt-4o"

        self.llm = ChatOpenAI(
            base_url=base_url,
            model=model_name,
            api_key=api_key,
        )
        if video_urls is None:
            self.video_urls = ['https://www.youtube.com/watch?v=2uYu8nMR5O4',
                                'https://www.youtube.com/watch?v=vuFpTFwEBaQ',
                                'https://www.youtube.com/watch?v=n6aLX9oETic',
                                'https://www.youtube.com/watch?v=vxrnCzjDJU4',
                                'https://www.youtube.com/watch?v=Yo7WvXjFHSA',
                                'https://www.youtube.com/watch?v=sJE5i-co2y0',
                                'https://www.youtube.com/watch?v=C8w855EfKL4',
                                'https://www.youtube.com/watch?v=NzwlDmMkEKs',
                                'https://www.youtube.com/watch?v=ll27QTDv2GQ',
                                'https://www.youtube.com/watch?v=MN3RJWcJKnk',]
        else: 
           self.video_urls = video_urls
        self.chunks_root_path = chunks_root_path

            

    def encode_np_array_to_base64(self, frame, img_format='jpg'):

        success, buffer = cv2.imencode(f".{img_format}", frame)
        if not success:
            raise ValueError("Could not encode frame as image.")

        base64_image = base64.b64encode(buffer).decode("utf-8")
        return base64_image
    def get_image_discription(self, frame):

      base64_image = self.encode_np_array_to_base64(frame, img_format='jpg')
    
      messages = [
          {
              "role": "system",
              "content": '''You are an AI assistant responsible for analyzing frames and generating comprehensive descriptions based on the following aspects:

    1. **OCR Results**: Automatically extract and describe any visible text from the frame.
    2. **Object Detection**: Identify objects in the frame, their positions, and labels.
    3. **Scene Context**: Infer the overall scene based on object interactions, environmental elements, and visual cues from the frame.
    
    Your task is to generate detailed and structured descriptions that can be used to create questions later. Each description must be accurate, detailed, and organized to support question generation based on the visual content.
    
    Structure your output as follows:
    - **Overall Description**: Provide a final, cohesive description that captures the essence of the frame, considering all objects, text, and environmental details.
    
    Be clear, concise, and ensure your description can serve as a future prompt for generating questions about the frame.''',
          },
          {
              "role": "user",
              "content": [
                  {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpg;base64,{base64_image}",
                          "detail": "auto",
                      },
                  },
              ],
          },
      ]
    
      ai_message = self.llm.invoke(messages)
      return ai_message.content
    
    def generate_QAs(self, description):

      messages = [
          {
              "role": "system",
              "content": '''*You are an AI assistant tasked with generating both questions and answers from a provided video description. Your goal is to create up to 50 unique, relevant question-answer pairs based on the content of the video, treating these as exam-style questions on the material presented. Ensure the questions require retrieval-augmented support for accurate answering, aiming to create questions that standard language models like GPT-4 or LLaMA3 would struggle to answer solely from pre-trained knowledge.*

    - **Generate Answers**: Provide a detailed, accurate answer for each question, formatted as follows:

    1. **question** : [Your question here]
    **answer** : [Your answer here]

    2. **question** : [Your question here]
    **answer** : [Your answer here]

    - **Content-Focused Questions**: Cover various aspects of the video’s core content, avoiding any details about the video format, duration, etc.
    - **Avoid Redundancy**: Make sure each question is unique, not overly similar, and diverse in focus.
    - **Clarity and Structure**: Ensure questions are clear, detailed, and structured to support retrieval and answer generation within a Retrieval-Augmented Generation (RAG) system.

    If generating 50 unique question-answer pairs is not feasible without repetition, stop once you have exhausted all possible unique variations.''',
          },
          {
              "role": "user",
              "content": [
                {
                    "type": "text",
                    "text": f'''Here is the description of the video:

    {description}

    Generate up to 50 unique questions based on this description. Ensure the questions cover different aspects of the video content and avoid duplication. If 50 questions cannot be made, generate as many as possible.
    ''',
                },
              ],
          },
      ]

      ai_message = self.llm.invoke(messages)
      return ai_message.content

    def download_video(self, url):
        yt = YouTube(url, on_progress_callback = on_progress)
        ys = yt.streams.get_highest_resolution()
        file_path = ys.download()
        file_name = os.path.basename(file_path)
        return file_path, file_name
    
    def get_transcript(self, video_id):
      transcript = YouTubeTranscriptApi.get_transcript(video_id)

      full_transcript = " ".join([entry['text'] for entry in transcript])
      return full_transcript
    
    def extract_video_id(self, url):
      video_id = url.split('v=')[1]
      ampersand_position = video_id.find('&')
      if ampersand_position != -1:
          video_id = video_id[:ampersand_position]
      return video_id

    def get_raw_transcripts(self, video_id):
      
      return YouTubeTranscriptApi.get_transcript(video_id)
    
    
    def analize_frames(self, frames):
      results = []
      for i, frame in enumerate(frames):
        llm_result = self.get_image_discription(frame)
        results.append(f"Frame {i}: {llm_result}")
      return "\n".join(results)

    def get_combined_results(self, frames, transcript):

        analized_frames_results = self.analize_frames(frames)

        combined_results = f"Transcript:\n{transcript}\n\nAnalize Frames Results:\n{analized_frames_results}"

        return combined_results
    

    def save_as_json(self, idx, video_id, video_name, result, video_url, file_path = "Q&A.json"):

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as fp:
               metadata = json.load(fp)
        else:
           metadata = [] 

        dict_template = {'video_id' : video_id,}
        dict_template = {'video_url' : video_url,}
        dict_template = {'video_name' : video_name,}



        sub_dict_template = {}
        QAs = result.split('\n\n')
        for i, QA in enumerate (QAs):
            Q = ''
            A = ''
            splited_QAs = QA.split('\n')

            index1 = splited_QAs[0].find(': ')
            Q = splited_QAs[0][index1 + len(': '):].strip()

            index2 = splited_QAs[1].find(': ')
            A = splited_QAs[1][index2 + len(': '):].strip()

            sub_dict_template[f'question {i}'] = Q
            sub_dict_template[f'answer {i}'] = A
        dict_template['Q&A'] = sub_dict_template
        metadata.append(dict_template)

        with open("Q&A.json", "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=4)

        return dict_template

    
    def get_llm_credit(self):
        url = "https://api.avalai.ir/user/credit"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer aa-pVNCqkgpuqSpvC8EdFmr8tzJy9HhzEkz5OGfT8YWxw9pB7kK"
        }
        return requests.get(url, headers=headers)
    

    def get_video_details(self):

        video_details = []
       
        for video_url in tqdm(self.video_urls, ):

            video_id = self.extract_video_id(video_url)
            transcript = self.get_transcript(video_id)
            path = self.download_video(video_url)[0] 
            video_detail_dict = {}
            video_name = os.path.basename(path).split('.')[0]


            video_detail_dict['id'] = video_id
            video_detail_dict['url'] = video_url
            video_detail_dict['path'] = path
            video_detail_dict['transcript'] =  transcript
            video_detail_dict['video_name'] =  video_name

            video_details.append(video_detail_dict)

        return video_details
    def get_chunk_frames(self, chunks_path):

        frames = []
        for file in os.listdir(chunks_path):
            full_path = os.path.join(chunks_path, file)
            if os.path.isfile(full_path) and file.lower().endswith(".jpg"):
                image = cv2.imread(full_path)
                if image is not None:
                    frames.append(image)
        return frames
    
    def process_videos(self):
        
        video_details = self.get_video_details()

        for i, video_detail in enumerate(tqdm(video_details,)):

            chunks_path =  os.path.join(self.chunks_root_path, video_details['video_name'].translate(str.maketrans('\\/:*?"<>|', '_' * 9)))
            chunks_path =  os.path.join(chunks_path, 'hybrid_clip_ssim_frame_dir')

            frames = self.get_chunk_frames(chunks_path)
            transcript = video_detail['transcript']

            video_desc = self.get_combined_results(frames, transcript)

            QAs = self.generate_QAs(video_desc)
            self.save_as_json(i+1, video_details['id'],video_details['video_name'] , QAs, video_detail['url'])
    
    def prepare_concatenation_with_metadata(self, transcripts):
        concatenated_text = ""
        transcript_metadata = []  # To store positions and timestamps
        current_pos = 0

        for item in transcripts:
            text = item["text"]
            start_time = item["start"]
            duration = item["duration"]
            concatenated_text += text + " "  # Add text with a space separator
            transcript_metadata.append({
                "start_pos": current_pos,
                "end_pos": current_pos + len(text),
                "start_time": start_time,
                "end_time": start_time + duration
            })

            current_pos += len(text) + 1  # Update for the space

        return concatenated_text.strip(), transcript_metadata

    def get_videos_transcript(self):

        video_details = self.get_video_details()

        for video_detail in tqdm(video_details,):

            transcripts_dir = os.path.join("./transcripts", video_detail['video_name'])
            
            os.makedirs(transcripts_dir, exist_ok=True)
            
            transcripts = self.get_raw_transcripts(video_detail['id'])

            full_transcript, transcript_metadata = self.prepare_concatenation_with_metadata(transcripts)

            transcript_txt_path = os.path.join(transcripts_dir, "transcript.txt")
            metadata_json_path = os.path.join(transcripts_dir, "metadata.json")
            
                    # Save full transcript to a text file
            with open(transcript_txt_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(full_transcript)
            # Save metadata to a JSON file
            with open(metadata_json_path, "w", encoding="utf-8") as json_file:
                json.dump(transcript_metadata, json_file, indent=4, ensure_ascii=False)