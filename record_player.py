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
    parser = argparse.ArgumentParser(description='Record player')
    parser.add_argument('--record-file', metavar='FILE',
                        type=str, nargs='+',
                        help='Input record files to play')
    parser.add_argument('--record-folder', metavar='FOLDER',
                        type=str, nargs='+',
                        help='Input record folder to play')

    args = parser.parse_args()

    if not any([args.record_file, args.record_folder]):
        parser.print_help()
        exit(0)

    record_file_to_play = []
    if args.record_file:
        record_file_to_play.extend(args.record_file)

    if args.record_folder:
        for folder in args.record_folder:
            files = glob.glob(os.path.join(folder, '*.json'))
            files = sorted(files)
            record_file_to_play.extend(files)

    for record_file in record_file_to_play:
        print ('Play record file: %s' % record_file)
        record_json = {}
        with open(record_file) as f:
            record_json = json.load(f)

        for record in record_json['records']:
            img = cv2.cvtColor(np.asarray(Image.open(
                BytesIO(base64.b64decode(record["image"])))), cv2.COLOR_RGB2BGR)
            curr_speed = record['curr_speed']
            curr_steering_angle = record['curr_steering_angle']
            cmd_speed = record['cmd_speed']
            cmd_steering_angle = record['cmd_steering_angle']

            cv2.putText(
                img, 'CMD. Spd.: %.2f, CMD. StrAgl.: %.2f' % (cmd_speed, cmd_steering_angle),
                (10, 12), cv2.FONT_ITALIC, 0.4, (0, 0, 255), 1)
            cv2.putText(
                img, 'CURR. Spd.: %.2f, CURR. StrAgl.: %.2f' % (curr_speed, curr_steering_angle),
                (10, 30), cv2.FONT_ITALIC, 0.4, (255, 0, 0), 1)

            cv2.imshow('image', img)
            if cv2.waitKey(50) == 27:  # ESC to exit
                exit(0)
