import os
import requests
import base64
import json
import shutil
import time
from pydub import AudioSegment
from pydub.utils import which


class BaiduVoiceToTxt():
    # 初始化函数
    def __init__(self):
        # 定义要进行切割的pcm文件的位置。speech-vad-demo固定好的，没的选
        self.pcm_path = ".\\speech-vad-demo\\pcm\\16k_1.pcm"
        # 定义pcm文件被切割后，分割成的文件输出到的目录。speech-vad-demo固定好的，没的选
        self.output_pcm_path = ".\\speech-vad-demo\\output_pcm\\"
        # 手动告诉 pydub ffmpeg 的位置
        ffmpeg_path = r"D:\Desktop\Vedio-Summarizer\Vedio-Summarizer\ffmpeg\bin\ffmpeg.exe"
        AudioSegment.converter = ffmpeg_path
        print(f"FFmpeg path set for pydub: {AudioSegment.converter}")

    # 百度AI接口只接受pcm格式，所以需要转换格式
    # 此函数用于将要识别的mp3文件转换成pcm格式，并输出为.\speech-vad-demo\pcm\16k_1.pcm
    def change_file_format(self, filepath):
        file_name = filepath
        # 如果.\speech-vad-demo\pcm\16k_1.pcm文件已存在，则先将其删除
        if os.path.isfile(f"{self.pcm_path}"):
            os.remove(f"{self.pcm_path}")
        ffmpeg_path = r"D:\Desktop\Vedio-Summarizer\Vedio-Summarizer\ffmpeg\bin\ffmpeg.exe"
        # 调用系统命令，将文件转换成pcm格式，并输出为.\speech-vad-demo\pcm\16k_1.pcm
        change_file_format_command = f"{ffmpeg_path} -y -i {file_name} -acodec pcm_s16le -f s16le -ac 1 -ar 16000 {self.pcm_path}"
        os.system(change_file_format_command)

    
    # 此函数用于将.\speech-vad-demo\pcm\16k_1.pcm切割
    def devide_video(self):
        # 如果切割输出目录.\speech-vad-demo\output_pcm\已存在，那其中很可能已有文件，先将其清空
        # 清空目录的文件是先删除，再创建
        if os.path.isdir(f"{self.output_pcm_path}"):
            shutil.rmtree(f"{self.output_pcm_path}")
        time.sleep(1)
        os.mkdir(f"{self.output_pcm_path}")
        # vad-demo.exe使用相对路径.\pcm和.\output_pcm，所以先要将当前工作目录切换到.\speech-vad-demo下不然vad-demo.exe找不到文件
        os.chdir(".\\speech-vad-demo\\")
        # 直接执行.\vad-demo.exe，其默认会将.\pcm\16k_1.pcm文件切割并输出到.\output_pcm目录下
        devide_video_command = ".\\vad-demo.exe"
        os.system(devide_video_command)

        os.chdir("..\\")


    # 此函数用于申请访问ai接口的access_token
    def get_access_token(self):
        client_id = 'zCYIl0xh1BgqEgLTj9nul4vD'
        client_secret = 'P7VwAVVemI508eOeafEQoL7io592wlOv'
        auth_url = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=' + client_id + '&client_secret=' + client_secret

        response_at = requests.get(auth_url)
        # 以json格式读取响应结果
        json_result = json.loads(response_at.text)
        # 获取access_token
        access_token = json_result['access_token']
        return access_token

    # 此函数用于将.\speech-vad-demo\output_pcm\下的单个文件由语音转成文件
    def transfer_voice_to_srt(self, access_token, filepath):
        # 百度语音识别接口
        url_voice_ident = "http://vop.baidu.com/server_api"
        # 接口规范，以json格式post数据
        headers = {
            'Content-Type': 'application/json'
        }
        # 打开pcm文件并读取文件内容
        pcm_obj = open(filepath, 'rb')
        pcm_content_base64 = base64.b64encode(pcm_obj.read())
        pcm_obj.close()
        # 获取pcm文件大小
        pcm_content_len = os.path.getsize(filepath)

        post_data = {
            "format": "pcm",
            "rate": 16000,
            "dev_pid": 1737,
            "channel": 1,
            "token": access_token,
            "cuid": "1111111111",
            "len": pcm_content_len,
            "speech": pcm_content_base64.decode(),
        }
        proxies = {
            'http': "127.0.0.1:8080"
        }
        # 调用接口，进行音文转换
        response = requests.post(url_voice_ident, headers=headers, data=json.dumps(post_data))
        # response = requests.post(url_voice_ident,headers=headers,data=json.dumps(post_data),proxies=proxies)
        return response.text


def transcribe_audio(audio_path):
    bvt = BaiduVoiceToTxt()
    bvt.change_file_format(audio_path)
    access_token = bvt.get_access_token()
    bvt.devide_video()  # 修改这里
    recognized_text = ""
    for root, dirs, files in os.walk(bvt.output_pcm_path):
        for file in files:
            file_path = os.path.join(root, file)
            response_text = bvt.transfer_voice_to_srt(access_token, file_path)  # 修改这里
            json_result = json.loads(response_text)
            if json_result['err_no'] == 0:
                recognized_text += json_result['result'][0].strip()
    print(f"Recognized text: {recognized_text}")
    return recognized_text




