import uuid

import pytest

import facefusion.globals
from facefusion.face_analyser import pre_check, clear_face_analyser, get_many_faces, compare_faces
from facefusion.filesystem import is_image, is_video
from facefusion.vision import read_static_image, write_image, normalize_frame_color, count_video_frame_total, \
	get_video_frame


@pytest.fixture(autouse=True)
def before_each() -> None:
	facefusion.globals.face_detector_score = 0.5
	facefusion.globals.face_landmarker_score = 0.5
	facefusion.globals.face_recognizer_model = 'arcface_inswapper'
	facefusion.globals.reference_face_distance = 0.6
	clear_face_analyser()


def test_face_detector() -> None:
	facefusion.globals.face_detector_model = 'yoloface'
	facefusion.globals.face_detector_size = '640x640'

	pre_check()
	source_paths = \
		[
			'.assets/examples/source.png',
			'.assets/examples/source.mp4'
		]
	for source_path in source_paths:
		face_frames: [dict] = []
		if is_image(source_path):
			source_frame = read_static_image(source_path)
			faces = get_many_faces(source_frame)
			for face in faces:
				face_frames.append({
					"face": face,
					"frame": source_frame
				})
		if is_video(source_path):
			n_frame = count_video_frame_total(source_path)
			for i in range(0, min(10, n_frame)):
				source_frame = get_video_frame(source_path, i)
				faces = get_many_faces(source_frame)
				for face in faces:
					face_frames.append({
						"face": face,
						"frame": source_frame
					})
		diff_face_frames = []
		for face_frame in face_frames:
			existed = False
			for diff_face_frame in diff_face_frames:
				if compare_faces(face_frame["face"], diff_face_frame["face"],
								 facefusion.globals.reference_face_distance):
					existed = True
					break
			if not existed:
				diff_face_frames.append(face_frame)
		for diff_face_frame in diff_face_frames:
			face = diff_face_frame["face"]
			frame = diff_face_frame["frame"]
			start_x, start_y, end_x, end_y = map(int, face.bounding_box)
			padding_x = int((end_x - start_x) * 0.25)
			padding_y = int((end_y - start_y) * 0.25)
			start_x = max(0, start_x - padding_x)
			start_y = max(0, start_y - padding_y)
			end_x = max(0, end_x + padding_x)
			end_y = max(0, end_y + padding_y)
			crop_vision_frame = frame[start_y:end_y, start_x:end_x]
			crop_vision_frame = normalize_frame_color(crop_vision_frame)
			if "source.png" in source_path:
				target_path = source_path.replace('source.png', 'target') + "-image-" + str(uuid.uuid4()) + ".png"
			else:
				target_path = source_path.replace('source.mp4', 'target') + "-video-" + str(uuid.uuid4()) + ".png"
			write_image(target_path, crop_vision_frame)
