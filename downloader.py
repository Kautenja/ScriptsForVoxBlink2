"""A script to download VoxBlink2."""
import argparse
import os
import sys
from tqdm import tqdm
import yt_dlp
from yt_dlp.utils import download_range_func
from parallel_map import parallel_map


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('base_dir',
    type=str,
    help="The output directory to save video/audio files to.",
    default='videos',
)
parser.add_argument('--mode',
    type=str,
    help='"video" to download videos, "audio" to download only audio.',
    choices={'video', 'audio'},
    default='video',
)
parser.add_argument('--audio_sample_rate',
    type=int,
    help='The sample rate for downloaded audio.',
    default=16000,
    choices={8000, 16000, 22050, 44100, 48000, 96000, 192000},
)
args = parser.parse_args()


def make_speaker_dir(speaker_id: str):
    """
    Create a directory for the given speaker.

    Args:
        speaker_id: The ID of the speaker to create a directory for.

    """
    os.makedirs(os.path.join(args.base_dir, speaker_id), exist_ok=True)


def media_requires_download(data: tuple) -> bool:
    """
    Return True if the media requires a download or False if it exists on disk.

    Args:
        data: The tuple of (speaker_id, video_id) to check the existence of

    Returns:
        True if the media is not found on disk, False otherwise.

    """
    speaker_id, video_id = data
    if args.mode == 'audio':
        ext = 'wav'
    elif args.mode == 'video':
        ext = 'mp4'
    video_path = os.path.join(args.base_dir, speaker_id, f'{video_id}.{ext}')
    return not os.path.exists(video_path)


def download_video(data: tuple) -> int:
    """
    Download a video for a speaker/video pair.

    Args:
        data: The tuple of (speaker_id, video_id) to download.

    Returns:
        The yt-dlp error code.

    """
    speaker_id, video_id = data
    # if os.path.exists(os.path.join(args.base_dir, speaker_id, f'{video_id}.mp4')):
    #     return 0
    with yt_dlp.YoutubeDL({
        'format': 'bestvideo[height<=720]+bestaudio',
        'outtmpl': os.path.join(args.base_dir, speaker_id, '%(id)s.%(ext)s'),
        'noplaylist': True,
        'ignoreerrors': True,
        'max_sleep_interval': 0.2,
        'verbose': False,
        'quiet': True,
        'download_ranges': download_range_func(None, [(0, 60)]),
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        },],
        'postprocessor_args': [
            '-ar', str(args.audio_sample_rate),
            '-strict', '-2',
            '-async','1', '-r' ,'25'
        ],
    }) as ydl:
        return ydl.download(video_id)


def download_audio(data: tuple) -> int:
    """
    Download audio for a speaker/video pair.

    Args:
        data: The tuple of (speaker_id, video_id) to download.

    Returns:
        The yt-dlp error code.

    """
    speaker_id, video_id = data
    # if os.path.exists(os.path.join(args.base_dir, speaker_id, f'{video_id}.wav')):
    #     return 0
    with yt_dlp.YoutubeDL({
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(args.base_dir, speaker_id, '%(id)s.%(ext)s'),
        'noplaylist': True,
        'ignoreerrors': True,
        'max_sleep_interval': 0.2,
        'verbose': False,
        'quiet': True,
        'download_ranges': download_range_func(None, [(0, 60)]),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192'
        },],
        'postprocessor_args': [
            '-ar', str(args.audio_sample_rate),
        ],
        'prefer_ffmpeg': True,
    }) as ydl:
        return ydl.download(video_id)


if __name__ == '__main__':
    spk2videos_file_path = os.path.join(PACKAGE_DIR, 'data', 'spk2videos')
    if not os.path.exists(spk2videos_file_path):
        print(f'Did not find video list at {spk2videos_file_path}')
        sys.exit()
    # Load Videos
    with open(spk2videos_file_path, 'r') as file:
        spk2videos = {l.split()[0]: l.strip().split()[1:] for l in file}
    print(f'Found {len(spk2videos)} speakers')
    # Count the number of videos per user.
    video_counts = []
    for _, videos in spk2videos.items():
        video_counts.append(len(videos))
    print(f'Found {sum(video_counts)} videos')
    print(f'Users have on average {sum(video_counts) / len(video_counts):.2f} videos')
    print(f'Users have at least {min(video_counts)} videos')
    print(f'Users have at most {max(video_counts)} videos')
    # Make directories for all of the speakers in the dataset.
    # print('Preemptively creating output speaker directories')
    # parallel_map(make_speaker_dir, spk2videos, desc='Creating speaker directories')
    # Calculate the paths for the videos to download.
    print('Generating speaker/video pairs for downloading')
    workload = [(key, value) for key, values in spk2videos.items() for value in values]
    workload = list(sorted(tqdm(workload, desc='Sorting workload')))
    workload = list(filter(media_requires_download, tqdm(workload, desc='Filtering for missing files')))
    # requires_download = parallel_map(media_requires_download, workload, desc='Finding missing files')
    print(f'Found {len(workload)} files that require downloading')
    print('Dispatching parallel download operations')
    if args.mode == 'audio':
        parallel_map(download_audio, workload)
    elif args.mode == 'video':
        parallel_map(download_video, workload)
