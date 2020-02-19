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
from typing import List

import jetson.inference
import jetson.utils

from pydantic import BaseModel, ValidationError, Schema
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
import queue
import concurrent.futures
import time
import threading
import logging

UPLOAD_FRAME_WITH_DETECTIONS = True
API_KEY = "abb6a29f-d60c-499f-a8e9-57a1af56c1a0"
CROP_COUNT_LOCK = threading.Lock()
CROP_COUNT = 0


class CropImage:
    def __init__(self, data, crop_id, camera_id, timestamp, location):
        self.data = data
        self.crop_id = crop_id
        self.camera_id = camera_id
        self.timestamp = timestamp
        self.location = location


class Location(BaseModel):
    latitude: float = Schema(..., example=50.431782)
    longitude: float = Schema(..., example=30.516382)


class CropMetadata(BaseModel):
    crop_id: int = Schema(..., description="User defined crop id", example=7564)
    bbox_top: int = Schema(..., description="Bounding box top coordinate", example=127)
    bbox_left: int = Schema(..., description="Bounding box left coordinate", example=128)
    bbox_width: int = Schema(..., description="Bounding box width", example=129)
    bbox_height: int = Schema(..., description="Bounding box height", example=130)
    person_id: str = Schema(..., description="Person unique identifier", example=1)
    confidence_level: float = Schema(..., description="Confidence level of detection", example=0.87)


class UploadCropsRequest(BaseModel):
    crops: List[CropMetadata] = Schema(..., description="User defined array crop metadata",
                                       example=[{"crop_id": 7564, "bbox_top": 127, "bbox_left": 128,
                                                 "bbox_width": 129, "bbox_height": 130, "confidence_level": 0.87},
                                                {"crop_id": 7565, "bbox_top": 200, "bbox_left": 150,
                                                 "bbox_width": 84, "bbox_height": 92, "confidence_level": 0.77}
                                                ])
    camera_id: str = Schema(..., description="Unique camera id", example='12b8f8d2-3b28-4e8a-a47e-d927352f915e')
    timestamp: float = Schema(..., description="UNIX timestamp", example=1568140822.407)
    location: Location = Schema(..., description="Geo location")
    tags: List[str] = Schema(..., description="Camera tags",
                             example=['building_id:621017CA-3717-4674-BFDD-70B11B53C324',
                                      'front-door',
                                      '621017CA-3717-4674-BFDD-70B11B53C324'])


def _get_crop_id(camera_id):
    with CROP_COUNT_LOCK:
        global CROP_COUNT
        crop_id = (camera_id << 24) + CROP_COUNT
        CROP_COUNT += 1

    return crop_id


def _convert_image(img, width, height):
    array = jetson.utils.cudaToNumpy(img, width, height, 4)
    buf = BytesIO()

    im = Image.fromarray(array.astype(numpy.uint8), "RGB")
    im.save(buf, format="jpeg")

    return buf.getvalue()


def crop(camera_id, img, width, height, detection):
    array = jetson.utils.cudaToNumpy(img, width, height, 4)
    buf = BytesIO()

    print(detection)
    im = Image.fromarray(array.astype(numpy.uint8), "RGBA").crop(
        (detection.Left, detection.Top, detection.Right, detection.Bottom))
    im.save(buf, format="png")

    crop_id = _get_crop_id(camera_id)
    c = CropImage(buf.getvalue(), crop_id, camera_id, time.time(), (0, 0))

    return c


async def upload_crop(session, img):
    data = aiohttp.FormData()
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

    response = await session.post('https://k1.traces.cloud/api/v1/uploadCropsWithMetadata',
                                  data=data, headers={'X-Traces-API-Key': API_KEY})

    if response.status != HTTPCreated.status_code:
        logging.error("Image upload failure. Request status code: " + str(response.status))
    logging.info("Image uploaded: " + str(img.crop_id))


async def upload_frame(session, img, request_data):
    data = aiohttp.FormData()
    data.add_field('metadata', request_data.json())
    data.add_field('image', img, filename='image.png')

    logging.info(f"Upload frame request: {request_data}")

    response = await session.post('https://k1.traces.cloud/api/v1/uploadCropsWithMetadata',
                                  data=data, headers={'X-Traces-API-Key': API_KEY})

    if response.status != HTTPCreated.status_code:
        logging.error("Image upload failure. Request status code: " + str(response.status))
    logging.info("Frame uploaded.")


async def main(opt, camera):
    event_loop = asyncio.get_event_loop()
    async with aiohttp.ClientSession() as session:
        while True:
            # capture the image
            img, width, height = camera.CaptureRGBA(zeroCopy=1)
            jetson.utils.cudaDeviceSynchronize()
            detections = net.Detect(img, width, height, opt.overlay)

            img = _convert_image(img, width, height)
            if len(detections):
                logging.info("detected {:d} objects in image".format(len(detections)))

                if UPLOAD_FRAME_WITH_DETECTIONS:
                    upload_request = UploadCropsRequest(**{
                        "crops": list(),
                        "camera_id": opt.camera_id,
                        "timestamp": time.time(),
                        "location": Location(**{
                            "latitude": 0,
                            "longitude": 0,
                        }),
                        "tags": ["test"],
                    })
                    for detection in detections:
                        c = CropMetadata(**{
                            "crop_id": _get_crop_id(opt.camera_id),
                            "bbox_top": detection.Top,
                            "bbox_left": detection.Left,
                            "bbox_width": detection.Right - detection.Left,
                            "bbox_height": detection.Bottom - detection.Top,
                            "person_id": "",
                            "confidence_level": 0.62,
                        })
                        upload_request.crops.append(c)

                    event_loop.create_task(upload_frame(session, img, upload_request))
                else:
                    for detection in detections:
                        c = crop(opt.camera_id, img, width, height, detection)
                        event_loop.create_task(upload_crop(session, c))

                await asyncio.sleep(1)


if __name__ == "__main__":
    # parse the command line
    parser = argparse.ArgumentParser(
        description="Locate objects in a live camera stream using an object detection DNN.",
        formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage())

    parser.add_argument("--network", type=str, default="ssd-mobilenet-v2",
                        help="pre-trained model to load (see below for options)")
    parser.add_argument("--overlay", type=str, default="box,labels,conf",
                        help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
    parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")
    parser.add_argument("--camera", type=str, default="0",
                        help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
    parser.add_argument("--width", type=int, default=1280,
                        help="desired width of camera stream (default is 1280 pixels)")
    parser.add_argument("--height", type=int, default=720,
                        help="desired height of camera stream (default is 720 pixels)")
    parser.add_argument("--camera_id", type=int, default=0, help="demo camera index")

    try:
        opt = parser.parse_known_args()[0]
    except:
        print("")
        parser.print_help()
        sys.exit(0)

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    # load the object detection network
    net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

    # create the camera and display
    camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)
    # executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

    loop = asyncio.get_event_loop()
    try:
        asyncio.ensure_future(main(opt, camera))
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # executor.shutdown(wait=False)
        asyncio.gather(*asyncio.Task.all_tasks())
        loop.close()
        logging.info("Execution canceled")
