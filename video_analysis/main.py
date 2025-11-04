# NOTE in general, I need some function or module that takes in data from both a video and a user

from pathlib import Path
import json
import pickle

from User import User
from Video import Video

import test_objects

from lexical_coverage import lexical_coverage
from phonetic_coverage import phonetic_coverage_by_phonemes, phonetic_coverage_by_words
from adjust_speech_rate import adjust_speech_rate
from translate import translate_text

### An example of the simplest formula for difficulty score: a rating for each factor (out of 100 perhaps?), 
# multiplied by coefficients that represent the factor's weight. So we also need to find a way to convert each
# raw measurement into a percentage rating. But the main objective of the new study is basically to find the
# coefficients (or fit a new formula to the actual data on the independent vs dependent variable graph)
# NOTE: not sure if the difficulty score needs to be relative to a user or an absolute score (relative to the
# language as a whole?)

# Coefficients to weigh the factors in the difficulty score
LEXICAL_WEIGHT = 0.4
PHONETIC_WEIGHT = 0.3

def analyze_video_difficulty(video: Video, user: User):
    # lexical factors
    lex_cov_raw = lexical_coverage(video, user)[0]
    lex_cov_inferencing = lexical_coverage(video, user)[1]

    # phonetic factors
    phon_cov_phonemes = phonetic_coverage_by_phonemes(video, user)
    phon_cov_words = phonetic_coverage_by_words(video, user)

    # calculate difficulty score
    difficulty_score = (LEXICAL_WEIGHT * lex_cov_raw) + (PHONETIC_WEIGHT * phon_cov_phonemes)

    return difficulty_score

def process_videos_in_folder(video_folder_path: str, audio_folder_path: str) -> list[Video]:
    """
    Process the videos in a folder. Return a list of video objects and save data of each video to a JSON file.
    """
    video_list = []

    # Path to your folder
    video_dir = Path(video_folder_path)

    # Define allowed video extensions (lowercase)
    video_exts = {".mp4", ".mov", ".avi", ".mkv"}

    i = 1
    # Iterate over matching files
    for video_path in video_dir.iterdir():
        if video_path.suffix.lower() in video_exts:
            path = str(video_path.resolve())  # absolute path
            video = Video(path, audio_folder_path)
            video_list.append(video)
            with open(f"video_analysis/sampled_json_subset/{video_path.stem}_data.json", "w", encoding="utf-8") as f:
                json.dump(video.to_dict(), f, ensure_ascii=False, indent=4)
            print(f"✅ Video {i}: {path} is processed.")
        else:
            print(f"✅ Video {i} is NOT processed.")
        i += 1

    return video_list
    

### the driver for the entire video analysis algorithm ###

if __name__ == "__main__":
    # video = test_objects.video_etymology
    video = Video(path="video_analysis/videos/etymology.mp4",
                  audio_folder="video_analysis/audios")
    # user = test_objects.user

    # # Displaying video and user data
    # print(video)
    # for seg in video.transcript["segments"]:
    #     print(seg, "\n")
    # print(video.transcript["info"])

    # video = test_objects.video_etymology
    # print(video.transcript_text)
    # for seg in video.transcript["segments"]:
    #     print(seg.text, "\n")
    
    # print(user)

    # Calculating lexical coverage
    # print(lexical_coverage(video, user))

    # Processing videos in a folder
    # video_subset = process_videos_in_folder("video_analysis/sampled_videos_subset", 
    #                                         "video_analysis/sampled_audios_subset")
    # for video in video_subset:
    #     print(video)

    # Modifying videos and saving them as new files
    # adjust speech rate
    # translation


    # Translating transcripts
    # translation = translate_text(video.transcript_text, to_lang=user.l1)
    # print(f"Translated Transcript: {translation}")

    # Displaying difficulty score (to be implemented)
    # difficulty_score = analyze_video_difficulty(video, user)
    # print(f"Video Difficulty Score: {difficulty_score}")

    # Ranking multiple videos
    # video_paths = ["video_analysis/video1.MP4", "video_analysis/video2.MP4"]
    # for path in video_paths:
    #     video = Video(path=path)
    #     difficulty_score = analyze_video_difficulty(video, user)
    #     print(f"Video: {path}, Difficulty Score: {difficulty_score}")
    # ranking = rank_videos(video_paths, user)
    # print(f"Video Rankings: {ranking}")

