
import wave
import pyaudio
import numpy as np
from pydub import AudioSegment

## 设置录音参数
CHUNK = 1024
FORMAT = pyaudio.paInt16  # 16-bit encoding
CHANNELS = 1  # mono
RATE = 16000  # sample rate
MAX_DURATION = 10  # maximum recording duration in seconds


def audio_record(out_file):
    p = pyaudio.PyAudio()
    # 创建音频流
    stream = p.open(format=FORMAT,  # 音频流wav格式
                    channels=CHANNELS,  # 单声道
                    rate=RATE,  # 采样率16000
                    input=True,
                    frames_per_buffer=CHUNK)

    print("开始录制...")

    frames = []  # 录制的音频流
    # 录制音频数据
    for i in range(0, int(RATE / CHUNK * MAX_DURATION)):
        data = stream.read(CHUNK)
        frames.append(data)
    # 录制完成
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 将录音数据保存到文件
    wf = wave.open(out_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # 将WAV文件转换为MP3
    sound = AudioSegment.from_wav(out_file)
    sound.export(out_file.replace(".wav", ".mp3"), format="mp3")

    return out_file.replace(".wav", ".mp3")
