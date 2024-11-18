import os
import json
import csv
import ffmpeg
import pandas as pd
import yt_dlp as ytdl
from pathlib import Path

# configuration
NUM_VIDEOS = 3
BASE_DIR = "./data"
AUDIO_DIR = os.path.join(BASE_DIR, "audio_files")
IMG_DIR = os.path.join(BASE_DIR, "rgb_frames")
CLIP_DIR = os.path.join(BASE_DIR, "10s_clips")
ANNOTATIONS_FILE = os.path.join(BASE_DIR, "train.csv")
# ONTOLOGY_JSON = "ontology.json"
FPS = 25
FRAME_SIZE = (224, 224)

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(CLIP_DIR, exist_ok=True)


def download_video(yt_id, start_seconds, end_seconds, save_path):
    try:
        ydl_opts = {
            "format": "best",
            "outtmpl": os.path.join(save_path, f"{yt_id}.mp4"),
        }

        with ytdl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(
                f"https://www.youtube.com/watch?v={yt_id}", download=True
            )
        video_path = os.path.join(save_path, f"{yt_id}.mp4")

        output_path = os.path.join(CLIP_DIR, f"{yt_id}_10s.mp4")

        ffmpeg.input(
            video_path, ss=start_seconds, t=end_seconds - start_seconds
        ).output(output_path).run(overwrite_output=True)

        os.remove(video_path)

        return output_path
    except ytdl.utils.ExtractorError as e:
        if "Video unavailable" in str(e):
            print(f"Video {yt_id} unavailable, skipping.")
        else:
            print(f"Error downloading video {yt_id}: {e}")
    except Exception as e:
        print(f"Unexpected error downloading video {yt_id}: {e}")
    return None


def extract_audio(video_path, audio_dir, yt_id):
    try:
        audio_path = os.path.join(audio_dir, f"{yt_id}.wav")
        ffmpeg.input(video_path).output(audio_path, ar=16000, ac=1).run(
            overwrite_output=True
        )
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None


def extract_frames(video_path, img_dir, yt_id):
    try:
        frame_dir = os.path.join(img_dir, yt_id)
        os.makedirs(frame_dir, exist_ok=True)
        output_pattern = os.path.join(frame_dir, "%04d.jpg")

        ffmpeg.input(video_path).output(
            output_pattern, vf=f"fps={FPS},scale={FRAME_SIZE[0]}:{FRAME_SIZE[1]}"
        ).run(overwrite_output=True)

        return frame_dir
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return None


def load_label_mapping(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


def process_annotations(input_csv, output_csv, label_mapping_file):
    with open(label_mapping_file, "r") as f:
        label_mapping = json.load(f)

    with open(input_csv, "r") as infile, open(output_csv, "w", newline="") as outfile:
        reader = csv.reader(infile, quotechar='"', skipinitialspace=True)
        writer = csv.writer(outfile)
        writer.writerow(["file_name", "label"])
        next(reader)

        for i, row in enumerate(reader):
            if i >= NUM_VIDEOS:
                break
            yt_id = row[0]
            labels = row[3]

            label_ids = labels.split(",")
            # print(labels)
            # print(label_ids)
            label_names = [
                label_mapping.get(label_id.strip(), "Unknown") for label_id in label_ids
            ]
            # print(label_names)
            writer.writerow([yt_id, ",".join(label_names)])


def main():
    label_mapping = load_label_mapping("label_mapping.json")
    dataset = pd.read_csv(
        "balanced_train_segments.csv",
        quotechar='"',
        on_bad_lines="skip",
        skipinitialspace=True,
    )

    for idx, row in dataset.iterrows():
        if idx >= NUM_VIDEOS:
            break

        yt_id, start, end, labels = row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3]

        print(f"Processing video {yt_id} ({idx + 1}/{NUM_VIDEOS})")
        print(f"Downloading video with yt_id: {yt_id}")
        video_path = download_video(yt_id, start, end, BASE_DIR)
        if not video_path:
            print(f"Error downloading video {yt_id}")
            continue

        extract_audio(video_path, AUDIO_DIR, yt_id)
        extract_frames(video_path, IMG_DIR, yt_id)

    process_annotations(
        "balanced_train_segments.csv",
        ANNOTATIONS_FILE,
        "label_mapping.json",
    )


if __name__ == "__main__":
    main()
