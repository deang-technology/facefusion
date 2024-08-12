FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /home/ubuntu/facefusion

RUN apt-get update
RUN apt-get install python3.10 -y
RUN apt-get install python-is-python3 -y
RUN apt-get install pip -y
RUN apt-get install git -y
RUN apt-get install curl -y
RUN apt-get install ffmpeg -y

COPY . /home/ubuntu/facefusion/

RUN python install.py --onnxruntime cuda-11.8 --skip-conda

ENTRYPOINT python -m uvicorn app:app --host 0.0.0.0
