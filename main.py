import os
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import scenedetect
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import whisperx
import openai
from PIL import Image
import numpy as np
from openai import OpenAI

client = OpenAI(api_key="your openai api here")

# Step 1: Scene Detection
def scene_detection(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    return scene_list

# Step 2: Split Video into Multiple Clips
def split_video(video_path, scene_list, output_dir):
    clip_paths = []
    for i, scene in enumerate(scene_list):
        start, end = scene[0].get_seconds(), scene[1].get_seconds()
        clip_path = os.path.join(output_dir, f"clip_{i+1}.mp4")
        ffmpeg_extract_subclip(video_path, start, end, targetname=clip_path)
        clip_paths.append(clip_path)
    return clip_paths

# Step 3: Speech Recognition using WhisperX
def speech_recognition(video_path):
    try:
        model = whisperx.load_model("large", device='cuda')
        result = model.transcribe(video_path)
        return result["text"]
    except Exception as e:
        print(f"Error during ASR: {e}")
        return "No transcription."

# Step 4: Generate Clip-Level Video Descriptions
def generate_clip_descriptions(clip_paths, scenes, num_frames=10):
    descriptions = []
    for clip_path, scene in zip(clip_paths, scenes):
        clip = VideoFileClip(clip_path)
        frames = sample_frames(clip, num_frames)
        images = [Image.fromarray(frame) for frame in frames]
        # Convert frames to base64 for OpenAI Image API
        image_files = [convert_to_base64(image) for image in images]
        content = []
        prompt = ("You are an expert in understanding scene transitions based on visual features in a video. "
                  "For the given sequence of images per timestamp, identify different scenes in the video. "
                  "Generate pure audio description for each scene with time ranges.\n"
                  f"{scene}")

        content.append({"type": "text", "text": prompt})
        for image_file in image_files:
            content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_file}"},
                        })

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=1000,
        )
        descriptions.append(response.choices[0].message.content)
        
    return "".join(descriptions)

# Helper function to sample frames
def sample_frames(video, num_frames):
    total_frames = int(video.fps * video.duration)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = [video.get_frame(i / video.fps) for i in frame_indices]
    return frames

# Helper function to convert PIL image to base64
def convert_to_base64(image):
    from io import BytesIO
    import base64

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Step 5: Generate a Coherent Script using GPT-4
def generate_script(clip_descriptions):
    prompt = (
        "You are an expert at understanding audio descriptions of different scenes in a video. "
        "Generate full audio description of each scene with non-overlapping time ranges. "
        "Keep as many scenes as possible covering all time ranges. "
        "Use character names wherever possible in the audio descriptions. "
        "Keep the audio description for each time range within one short sentence.\n\n"
    )
    prompt += "\n\n".join(clip_descriptions)
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1000,
    )
    return response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    video_path = "yourvideo.mp4"
    output_dir = "clips"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Scene Detection
    scenes = scene_detection(video_path)
    print("Detected scenes:", scenes)

    # Step 2: Split Video into Multiple Clips
    clip_paths = split_video(video_path, scenes, output_dir)
    print("Video clips created:", clip_paths)

    # Step 3: Speech Recognition using WhisperX
    transcript = speech_recognition(video_path)
    print("Transcript:", transcript)

    # Step 4: Generate Clip Descriptions
    clip_descriptions = generate_clip_descriptions(clip_paths, scenes, 10)
    print("Clip Descriptions:", clip_descriptions)

    clip_descriptions = clip_descriptions + f'\n\n ASR:{transcript}'

    # Step 5: Generate a Coherent Script
    script = generate_script(clip_descriptions)
    print("Generated Script:", script)
