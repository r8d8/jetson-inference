#!/usr/bin/python
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

import argparse
import sys
from PIL import Image 
import tempfile
import numpy, cv2
import aiohttp 
import asyncio
import json
from io import BytesIO
from aiohttp.web import HTTPCreated
import multiprocessing
import concurrent.futures
import time

API_KEY = "abb6a29f-d60c-499f-a8e9-57a1af56c1a0" 

class CropImage:
	def __init__(self, data, crop_id, camera_id, timestamp, location):
		self.data = data
		self.crop_id = crop_id
		self.camera_id = camera_id 
		self.timestamp = timestamp 
		self.location = location

async def upload(session, image_queue):
	data = aiohttp.FormData()
	img = image_queue.get()
	data.add_field('metadata', json.dumps({
		'crop_id': img.crop_id,
		'camera_id': img.camera_id,
		'timestamp': img.timestamp,
		'location': {
			'latitude': img.location[0],
			'longitude': img.location[1],
		},
		'tags': ['aff'],
	}))
	data.add_field('image', img.data, filename='image.png')

	response = await session.post('http://3.16.160.221/api/v1/uploadCropWithMetadata',
										data=data, headers={'X-Traces-API-Key': API_KEY})
	
	if response.status != HTTPCreated.status_code:
		print(">>> DEBUG: image upload failure. Request status code: ", response.status)

def crop(camera_id, img, width, height, detection, image_queue):
	array = jetson.utils.cudaToNumpy(img, width, height, 4)	
	buf = BytesIO()	
	
	print(detection)    
	im = Image.fromarray(array.astype(numpy.uint8), "RGBA").crop((detection.Left, detection.Top, detection.Right, detection.Bottom))
	im.save(buf, format="png")

	crop = CropImage(buf.getvalue(), 1, camera_id, time.time(), (0, 0))

	image_queue.put(crop)
		

async def main():
	# parse the command line
	parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
							formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage())

	parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)") 
	parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
	parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 
	parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
	parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
	parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")

	try:
		opt = parser.parse_known_args()[0]
	except:
		print("")
		parser.print_help()
		sys.exit(0)

	# load the object detection network
	net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

	# create the camera and display
	camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)
	executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

	# process frames until user exits
	img_queue = multiprocessing.Queue()
	loop = asyncio.get_event_loop()
	async with aiohttp.ClientSession() as session:
		while True:
			# capture the image
			img, width, height = camera.CaptureRGBA(zeroCopy=1)
			jetson.utils.cudaDeviceSynchronize()
			detections = net.Detect(img, width, height, opt.overlay)
			if len(detections):
				print("detected {:d} objects in image".format(len(detections)))
				for detection in detections:
					executor.submit(crop(opt.camera, img, width, height, detection, img_queue))
					loop.create_task(upload(session, img_queue))
				
			net.PrintProfilerTimes()


if __name__  == "__main__":
	loop = asyncio.get_event_loop()
	loop.run_until_complete(main())

