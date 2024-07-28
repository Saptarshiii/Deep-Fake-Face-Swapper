# Deep-Fake-Face-Swap
This Repo can change realilty with use of AI
Find the Demo of this repo here: https://drive.google.com/file/d/1QUvOh3V3RXT_oiaZ976Y4oGWZzMnECqr/view?usp=drive_link

## Pre-Requisite
1. CUDA
2. CUDNN
3. NIVIDIA Graphics Driver
4. Download the faceswapper model weights from this link: link is removed for security purposes

contact owner at: saptarshibanik123@gmail.com

Note: All of the above compatible with the user's coding environment and python libraries

## How to Use

Install all dependencies by running the following command
```
pip install -r requirments.txt
```
# For live video face swapping

for custom face swapping save the image source as source.jpg

run the **main.py** file
```
python .\main.py
```

# For face swapping a video file 
By default the codec is set to MPRV, output and input will work best with .MP4 files 
You will also need to adjust the output dimensions acc. to the input resolution of the video

set input video as video.mp4
set source image as source.jpg

run the **vid.py** file
```
python .\main.py
```


