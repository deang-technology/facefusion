import json
import logging
import mimetypes
import os.path
import subprocess
import sys
import uuid

import fastapi.responses
import requests

directory = '/home/ubuntu/facefusion/output'

app = fastapi.FastAPI()

url = "https://api.ipify.org?format=json"
response = requests.get(url)
ip = response.json()["ip"]
logging.info(f"Ip: {ip}")

port = 8000


@app.get("/file")
def get_file(file_url: str):
	media_type, _ = mimetypes.guess_type(file_url)
	return fastapi.responses.FileResponse(file_url, media_type=media_type)


@app.get('/face-detect')
def face_detect(file_url: str):
	file_path = f"{directory}/{str(uuid.uuid4())}"
	output_path = f"{directory}/{str(uuid.uuid4())}"
	resp = requests.get(file_url)
	with open(file_path, "wb") as f:
		f.write(resp.content)
	commands = [sys.executable,
				'run.py',
				'--only-detector',
				'-o',
				output_path,
				'-t',
				file_path,
				'--reference-frame-number',
				'10',
				'--face-detector-output-directory',
				directory,
				'--execution-providers',
				'cuda']
	with open("nohup.out", "a") as log_file:
		run = subprocess.run(commands, stdout=log_file, stderr=log_file, text=True)
		if run.returncode != 0:
			return {"status": -1, "message": "Error"}
		with open(output_path, "r") as ret_file:
			data = json.load(ret_file)
		data = list(map(lambda x: f"http://{ip}:{port}/file?file_url={x}", data))
		return {"status": 1, "message": "Success", "data": data}


@app.get('/face-swap')
def face_swap(target_url: str, face_url: str, source_url: str, type: str):
	target_file = f"{directory}/{str(uuid.uuid4())}"
	face_file = f"{directory}/{str(uuid.uuid4())}"
	source_file = f"{directory}/{str(uuid.uuid4())}"
	resp1 = requests.get(target_url)
	with open(target_file, "wb") as f1:
		f1.write(resp1.content)
	resp2 = requests.get(face_url)
	with open(face_file, "wb") as f2:
		f2.write(resp2.content)
	resp3 = requests.get(source_url)
	with open(source_file, "wb") as f3:
		f3.write(resp3.content)
	output_file = ""
	if type == "video":
		output_file = f"{directory}/{str(uuid.uuid4())}.mp4"
	if type == "image":
		output_file = f"{directory}/{str(uuid.uuid4())}.png"
	commands = [sys.executable,
				'run.py',
				'--face-selector-mode',
				'reference',
				'--frame-processors',
				'face_swapper',
				'-s',
				source_file,
				'-t',
				target_file,
				'-o',
				output_file,
				'-f',
				face_file,
				'--headless',
				'--execution-providers',
				'cuda']
	with open("nohup.out", "a") as log_file:
		run = subprocess.run(commands, stdout=log_file, stderr=log_file, text=True)
		if run.returncode != 0 or not os.path.exists(output_file):
			return {"status": -1, "message": "Error"}
		return {"status": 1, "message": "Success", "data": f"http://{ip}:{port}/file?file_url={output_file}"}
