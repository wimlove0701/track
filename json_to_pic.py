import os
import glob
import json
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import argparse

import cv2

"""
Record schema:
{
    "lap": lap index
    "elapsed_time": elapsed time during this lap
    "records": [
        {
            "image": base64 image
            "curr_speed": current speed from dashboard. unit: m/s
            "curr_steering_angle": current steering angle from dashboard. unit: degree
            "cmd_speed": speed command from human player. unit: m/s
            "cmd_steering_angle": steering angle command. unit: degree
            "time": timestamp. unit: seconds
        }
        ... list of record
    ]
}
"""

if __name__ == "__main__":
    json_folder = 'D:\\github\\track_json\\Track1'
    img_folder = 'D:\\github\\track_json\\IMG1'
    json_files = []

    files = glob.glob(os.path.join(json_folder, '*.json'))
    files = sorted(files)
    json_files.extend(files)

    #print json_files

    for record_file in json_files:
        print ('Parse record file: %s' % record_file)
        record_json = {}
        with open(record_file) as f:
            record_json = json.load(f)

        for record in record_json['records']:

            img = cv2.cvtColor(np.asarray(Image.open(
                BytesIO(base64.b64decode(record["image"])))), cv2.COLOR_RGB2BGR)
            curr_speed          = record['curr_speed']
            curr_steering_angle = record['curr_steering_angle']
            curr_throttle       = record['curr_throttle']

            cmd_speed           = record['cmd_speed']
            cmd_steering_angle  = record['cmd_steering_angle']
            cmd_throttle        = record['cmd_throttle']
            time = record['time']

            filename = os.path.join(img_folder, record_file + '-' + str(time) + '.jpg')
            #cv2.imwrite(filename, img)


