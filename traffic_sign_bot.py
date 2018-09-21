#!env python
#
# Auto-driving Bot
#
# Revision:      v1.2
# Released Date: Aug 20, 2018
#

from time import time
from PIL  import Image
from io   import BytesIO

#import datetime
import os
import cv2
import math
import numpy as np
import base64
import logging
import tensorflow as tf

def logit(msg):
    print("%s" % msg)

EPS  = np.finfo('float32').eps

class TrafficSignClassifier:

    def __init__(self, input_shape=[20, 32], learning_rate=1e-4, verbose=False):

        # Placeholders
        input_h, input_w = input_shape
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, input_h, input_w, 3])  # input placeholder
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None])
        self.keep_prob = tf.placeholder(dtype=tf.float32)  # dropout keep probability

        self.n_classes      = 9              # TBD
        self.learning_rate  = learning_rate  # learning rate used in train step

        self._inference     = None
        self._loss          = None
        self._train_step    = None
        self._accuracy      = None
        self._summaries     = None

        self.inference
        self.loss
        self.train_step
        self.accuracy
        # self.summaries # todo add these

        self.sess = tf.Session()
        #tf.saved_model.loader.load(self.sess,[tf.saved_model.tag_constants.SERVING], "model")
        self.saver = tf.train.Saver()
        # Important: Set the model path to your own model
        #self.saver.restore(self.sess, './checkpoint/tsd_epoch_199.ckpt')

        if verbose:
            self.print_summary()

    @property
    def inference(self):
        if self._inference is None:
            with tf.variable_scope('inference'):

                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3)

                conv1_filters = 8
                conv1 = tf.layers.conv2d(self.x, conv1_filters, kernel_size=(3, 3), padding='same',
                                         activation=tf.nn.relu, kernel_regularizer=kernel_regularizer)
                pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(1, 1), padding='same')

                conv2_filters = 16
                conv2 = tf.layers.conv2d(pool1, conv2_filters, kernel_size=(3, 3), padding='same',
                                         activation=tf.nn.relu, kernel_regularizer=kernel_regularizer)
                pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(1, 1), padding='same')

                _, h, w, c = pool2.get_shape().as_list()
                pool2_flat = tf.reshape(pool2, shape=[-1, h * w * c])

                pool2_drop = tf.nn.dropout(pool2_flat, keep_prob=self.keep_prob)

                hidden_units = self.n_classes
                hidden = tf.layers.dense(pool2_drop, units=hidden_units, activation=tf.nn.relu)

                logits = tf.layers.dense(hidden, units=self.n_classes, activation=None)

                self._inference = tf.nn.softmax(logits)

        return self._inference

    @property
    def loss(self):
        if self._loss is None:
            with tf.variable_scope('loss'):
                predictions = self.inference
                targets_onehot = tf.one_hot(self.targets, depth=self.n_classes)
                self._loss = tf.reduce_mean(-tf.reduce_sum(targets_onehot * tf.log(predictions + EPS), reduction_indices=1))
        return self._loss

    @property
    def train_step(self):
        if self._train_step is None:
            with tf.variable_scope('training'):
                self._train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        return self._train_step

    @property
    def accuracy(self):
        if self._accuracy is None:
            with tf.variable_scope('accuracy'):
                correct_predictions = tf.equal(tf.argmax(self.inference, axis=1),
                                               tf.argmax(tf.one_hot(self.targets, depth=self.n_classes), axis=1))
                self._accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return self._accuracy

    @staticmethod
    def print_summary():
        def pretty_border():
            print('*' * 50)

        pretty_border()
        print('Classifier initialized.')

        trainable_variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        num_trainable_params = np.sum([np.prod(v.get_shape()) for v in trainable_variables])
        print('Number of trainable parameters: {}'.format(num_trainable_params))
        pretty_border()
    
    @staticmethod
    def preprocess(x):
        """
        Roughly center on zero and put in range [-1, 1]
        """
        x = np.float32(x) - np.mean(x)
        x /= x.max()
        return x
    
    def predict(self, input_images):
        """
        classify the given image
        """
        #start = time.time()
        input_images = self.preprocess(input_images)
        input_images = np.array(input_images)
        prediction = self.sess.run(fetches=self.inference, feed_dict={self.x: input_images, self.keep_prob: 1.})
        prediction = np.argmax(prediction, axis=1)  # from onehot vectors to labels
        #done = time.time()
        #elapsed = done - start
        #print('prediction, elapsed time: {}'.format(elapsed))

        return prediction

class PID:
    def __init__(self, Kp, Ki, Kd, max_integral, min_interval = 0.001, set_point = 0.0, last_time = None):
        self._Kp           = Kp
        self._Ki           = Ki
        self._Kd           = Kd
        self._min_interval = min_interval
        self._max_integral = max_integral

        self._set_point    = set_point
        self._last_time    = last_time if last_time is not None else time()
        self._p_value      = 0.0
        self._i_value      = 0.0
        self._d_value      = 0.0
        self._d_time       = 0.0
        self._d_error      = 0.0
        self._last_error   = 0.0
        self._output       = 0.0


    def update(self, cur_value, cur_time = None):
        if cur_time is None:
            cur_time = time()

        error   = self._set_point - cur_value
        d_time  = cur_time - self._last_time
        d_error = error - self._last_error

        if d_time >= self._min_interval:
            self._p_value   = error
            self._i_value   = min(max(error * d_time, -self._max_integral), self._max_integral)
            self._d_value   = d_error / d_time if d_time > 0 else 0.0
            self._output    = self._p_value * self._Kp + self._i_value * self._Ki + self._d_value * self._Kd

            self._d_time     = d_time
            self._d_error    = d_error
            self._last_time  = cur_time
            self._last_error = error

        return self._output

    def reset(self, last_time = None, set_point = 0.0):
        self._set_point    = set_point
        self._last_time    = last_time if last_time is not None else time()
        self._p_value      = 0.0
        self._i_value      = 0.0
        self._d_value      = 0.0
        self._d_time       = 0.0
        self._d_error      = 0.0
        self._last_error   = 0.0
        self._output       = 0.0

    def assign_set_point(self, set_point):
        self._set_point = set_point

    def get_set_point(self):
        return self._set_point

    def get_p_value(self):
        return self._p_value

    def get_i_value(self):
        return self._i_value

    def get_d_value(self):
        return self._d_value

    def get_delta_time(self):
        return self._d_time

    def get_delta_error(self):
        return self._d_error

    def get_last_error(self):
        return self._last_error

    def get_last_time(self):
        return self._last_time

    def get_output(self):
        return self._output


class ImageProcessor(object):
    @staticmethod
    def show_image(img, name = "image", scale = 1.0):
        if scale and scale != 1.0:
            img = cv2.resize(img, newsize, interpolation=cv2.INTER_CUBIC) 

        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name, img)
        cv2.waitKey(1)


    @staticmethod
    def save_image(folder, img, prefix = "img", suffix = ""):
        from datetime import datetime
        filename = "%s-%s%s.jpg" % (prefix, datetime.now().strftime('%Y%m%d-%H%M%S-%f'), suffix)
        cv2.imwrite(os.path.join(folder, filename), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


    @staticmethod
    def rad2deg(radius):
        return radius / np.pi * 180.0


    @staticmethod
    def deg2rad(degree):
        return degree / 180.0 * np.pi


    @staticmethod
    def bgr2rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    @staticmethod
    def _normalize_brightness(img):
        maximum = img.max()
        if maximum == 0:
            return img
        adjustment = min(255.0/img.max(), 3.0)
        normalized = np.clip(img * adjustment, 0, 255)
        normalized = np.array(normalized, dtype=np.uint8)
        return normalized


    @staticmethod
    def _flatten_rgb(img):
        r, g, b = cv2.split(img)
        r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
        g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
        b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
        y_filter = ((r >= 128) & (g >= 128) & (b < 100))

        r[y_filter], g[y_filter] = 255, 255
        b[np.invert(y_filter)] = 0

        b[b_filter], b[np.invert(b_filter)] = 255, 0
        r[r_filter], r[np.invert(r_filter)] = 255, 0
        g[g_filter], g[np.invert(g_filter)] = 255, 0

        flattened = cv2.merge((r, g, b))
        return flattened


    @staticmethod
    def _crop_image(img):
        bottom_half_ratios = (0.55, 1.0)
        bottom_half_slice  = slice(*(int(x * img.shape[0]) for x in bottom_half_ratios))
        bottom_half        = img[bottom_half_slice, :, :]
        return bottom_half


    @staticmethod
    def preprocess(img):
        img = ImageProcessor._crop_image(img)
        #img = ImageProcessor._normalize_brightness(img)
        img = ImageProcessor._flatten_rgb(img)
        return img


    @staticmethod
    def find_lines(img):
        grayed      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred     = cv2.GaussianBlur(grayed, (3, 3), 0)
        #edged      = cv2.Canny(blurred, 0, 150)

        sobel_x     = cv2.Sobel(blurred, cv2.CV_16S, 1, 0)
        sobel_y     = cv2.Sobel(blurred, cv2.CV_16S, 0, 1)
        sobel_abs_x = cv2.convertScaleAbs(sobel_x)
        sobel_abs_y = cv2.convertScaleAbs(sobel_y)
        edged       = cv2.addWeighted(sobel_abs_x, 0.5, sobel_abs_y, 0.5, 0)

        lines       = cv2.HoughLinesP(edged, 1, np.pi / 180, 10, 5, 5)
        return lines


    @staticmethod
    def _find_best_matched_line(thetaA0, thetaB0, tolerance, vectors, matched = None, start_index = 0):
        if matched is not None:
            matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
            matched_angle = abs(np.pi/2 - matched_thetaB)

        for i in xrange(start_index, len(vectors)):
            distance, length, thetaA, thetaB, coord = vectors[i]

            if (thetaA0 is None or abs(thetaA - thetaA0) <= tolerance) and \
               (thetaB0 is None or abs(thetaB - thetaB0) <= tolerance):
                
                if matched is None:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi/2 - matched_thetaB)
                    continue

                heading_angle = abs(np.pi/2 - thetaB)

                if heading_angle > matched_angle:
                    continue
                if heading_angle < matched_angle:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi/2 - matched_thetaB)
                    continue
                if distance < matched_distance:
                    continue
                if distance > matched_distance:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi/2 - matched_thetaB)
                    continue
                if length < matched_length:
                    continue
                if length > matched_length:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi/2 - matched_thetaB)
                    continue

        return matched


    @staticmethod
    def find_steering_angle_by_line(img, last_steering_angle, debug = True):
        steering_angle = 0.0
        lines          = ImageProcessor.find_lines(img)

        if lines is None:
            return steering_angle

        image_height = img.shape[0]
        image_width  = img.shape[1]
        camera_x     = image_width / 2
        camera_y     = image_height
        vectors      = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                thetaA   = math.atan2(abs(y2 - y1), (x2 - x1))
                thetaB1  = math.atan2(abs(y1 - camera_y), (x1 - camera_x))
                thetaB2  = math.atan2(abs(y2 - camera_y), (x2 - camera_x))
                thetaB   = thetaB1 if abs(np.pi/2 - thetaB1) < abs(np.pi/2 - thetaB2) else thetaB2

                length   = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                distance = min(math.sqrt((x1 - camera_x) ** 2 + (y1 - camera_y) ** 2),
                               math.sqrt((x2 - camera_x) ** 2 + (y2 - camera_y) ** 2))

                vectors.append((distance, length, thetaA, thetaB, (x1, y1, x2, y2)))

                if debug:
                    # draw the edges
                    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

        #the line of the shortest distance and longer length will be the first choice
        vectors.sort(lambda a, b: cmp(a[0], b[0]) if a[0] != b[0] else -cmp(a[1], b[1]))

        best = vectors[0]
        best_distance, best_length, best_thetaA, best_thetaB, best_coord = best
        tolerance = np.pi / 180.0 * 10.0

        best = ImageProcessor._find_best_matched_line(best_thetaA, None, tolerance, vectors, matched = best, start_index = 1)
        best_distance, best_length, best_thetaA, best_thetaB, best_coord = best

        if debug:
            #draw the best line
            cv2.line(img, best_coord[:2], best_coord[2:], (0, 255, 255), 2)

        if abs(best_thetaB - np.pi/2) <= tolerance and abs(best_thetaA - best_thetaB) >= np.pi/4:
            print "*** sharp turning"
            best_x1, best_y1, best_x2, best_y2 = best_coord
            f = lambda x: int(((float(best_y2) - float(best_y1)) / (float(best_x2) - float(best_x1)) * (x - float(best_x1))) + float(best_y1))
            left_x , left_y  = 0, f(0)
            right_x, right_y = image_width - 1, f(image_width - 1)

            if left_y < right_y:
                best_thetaC = math.atan2(abs(left_y - camera_y), (left_x - camera_x))

                if debug:
                    #draw the last possible line
                    cv2.line(img, (left_x, left_y), (camera_x, camera_y), (255, 128, 128), 2)
                    cv2.line(img, (left_x, left_y), (best_x1, best_y1), (255, 128, 128), 2)
            else:
                best_thetaC = math.atan2(abs(right_y - camera_y), (right_x - camera_x))

                if debug:
                    #draw the last possible line
                    cv2.line(img, (right_x, right_y), (camera_x, camera_y), (255, 128, 128), 2)
                    cv2.line(img, (right_x, right_y), (best_x1, best_y1), (255, 128, 128), 2)

            steering_angle = best_thetaC
        else:
            steering_angle = best_thetaB

        if (steering_angle - np.pi/2) * (last_steering_angle - np.pi/2) < 0:
            last = ImageProcessor._find_best_matched_line(None, last_steering_angle, tolerance, vectors)

            if last:
                last_distance, last_length, last_thetaA, last_thetaB, last_coord = last
                steering_angle = last_thetaB

                if debug:
                    #draw the last possible line
                    cv2.line(img, last_coord[:2], last_coord[2:], (255, 128, 128), 2)

        if debug:
            #draw the steering direction
            r = 60
            x = image_width / 2 + int(r * math.cos(steering_angle))
            y = image_height    - int(r * math.sin(steering_angle))
            cv2.line(img, (image_width / 2, image_height), (x, y), (255, 0, 255), 2)
            logit("line angle: %0.2f, steering angle: %0.2f, last steering angle: %0.2f" % (ImageProcessor.rad2deg(best_thetaA), ImageProcessor.rad2deg(np.pi/2-steering_angle), ImageProcessor.rad2deg(np.pi/2-last_steering_angle)))

        return (np.pi/2 - steering_angle)


    @staticmethod
    def find_steering_angle_by_color(img, last_steering_angle, debug = True):
        r, g, b      = cv2.split(img)
        image_height = img.shape[0]
        image_width  = img.shape[1]
        camera_x     = image_width / 2
        image_sample = slice(0, int(image_height * 0.2))
        sr, sg, sb   = r[image_sample, :], g[image_sample, :], b[image_sample, :]
        track_list   = [sr, sg, sb]
        tracks       = map(lambda x: len(x[x > 20]), [sr, sg, sb])
        tracks_seen  = filter(lambda y: y > 50, tracks)

        if len(tracks_seen) == 0:
            return 0.0

        maximum_color_idx = np.argmax(tracks, axis=None)
        _target = track_list[maximum_color_idx]
        _y, _x = np.where(_target == 255)
        px = np.mean(_x)
        steering_angle = math.atan2(image_height, (px - camera_x))

        if debug:
            #draw the steering direction
            r = 60
            x = image_width / 2 + int(r * math.cos(steering_angle))
            y = image_height    - int(r * math.sin(steering_angle))
            cv2.line(img, (image_width / 2, image_height), (x, y), (255, 0, 255), 2)
            logit("steering angle: %0.2f, last steering angle: %0.2f" % (ImageProcessor.rad2deg(steering_angle), ImageProcessor.rad2deg(np.pi/2-last_steering_angle)))

        return (np.pi/2 - steering_angle) * 2.0

    @staticmethod
    def locate_traffic_signs(img):
        """
        locate traffic signs. return 32x20 images and bounding boxes
        """
        # traffic signs are at upper part of the whole image
        upper_part = slice(0, 100)
        frame = img[upper_part, :]

        # mask all colors except red (traffic sign is in red)
        R_mask = cv2.inRange(frame, (-1, -1, 70), (60, 60, 256))
        target = cv2.bitwise_and(frame, frame, mask=R_mask)
        origin = target.copy()
    
        ret, threshed_img = cv2.threshold(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_BINARY)
        
        # find contours first
        image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        prev_x = 0
        prev_y = 0
        prev_w = 0
        prev_h = 0

        # find bounding box of traffic signs
        bounding_rects = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            bounding_rects.append([x, y, w, h])

        bounding_rects = sorted(bounding_rects,key=lambda x: x[0])

        # merge bounding boxes (one traffic sign could be separated into two boxes, need to merge)
        merged_bounding_boxes = []
        final_bounding_boxes = []
        for b in range(len(bounding_rects)):
            # get the bounding rect
            x, y, w, h = bounding_rects[b]
            if w > 5 and h > 5:
                if len(merged_bounding_boxes) > 0:
                    if x > merged_bounding_boxes[-1][0] + merged_bounding_boxes[-1][2] + 20:
                        merged_bounding_boxes.append([x, y, w, h])
                    else:
                        merged_bounding_boxes[-1][2] = x - merged_bounding_boxes[-1][0]  + w
                        merged_bounding_boxes[-1][1] = min(merged_bounding_boxes[-1][1], y)
                        merged_bounding_boxes[-1][3] = y - merged_bounding_boxes[-1][1]  + h
                else:
                    merged_bounding_boxes.append([x, y, w, h])
        
        # drop small bounding boxes 
        traffic_signs = []
        for b in range(len(merged_bounding_boxes)):
            x, y, w, h = merged_bounding_boxes[b]
            if w > 10 and h > 5:
                final_bounding_boxes.append(merged_bounding_boxes[b])
                traffic_signs.append(cv2.resize(origin[y:y+h, x:x+w], (32, 20)))
                     
        return traffic_signs, final_bounding_boxes

traffic_sign_texts = {0:'None', 1: 'Right fork', 2: 'Left fork', 3: 'Right turn', 4: 'Left turn', 
5: 'Left U turn', 6: 'Right U turn', 7: 'Left lane warning', 8: 'Right lane warning'}

class AutoDrive(object):
    STEERING_PID_Kp             = 0.3
    STEERING_PID_Ki             = 0.01
    STEERING_PID_Kd             = 0.1
    STEERING_PID_max_integral   = 10
    THROTTLE_PID_Kp             = 0.02
    THROTTLE_PID_Ki             = 0.005
    THROTTLE_PID_Kd             = 0.02
    THROTTLE_PID_max_integral   = 0.5
    MAX_STEERING_HISTORY        = 3
    MAX_THROTTLE_HISTORY        = 3
    DEFAULT_SPEED               = 1.0

    debug = False

    def __init__(self, car, record_folder = None):
        self._record_folder    = record_folder
        self._steering_pid     = PID(Kp=self.STEERING_PID_Kp  , Ki=self.STEERING_PID_Ki  , Kd=self.STEERING_PID_Kd  , max_integral=self.STEERING_PID_max_integral  )
        self._throttle_pid     = PID(Kp=self.THROTTLE_PID_Kp  , Ki=self.THROTTLE_PID_Ki  , Kd=self.THROTTLE_PID_Kd  , max_integral=self.THROTTLE_PID_max_integral  )
        self._throttle_pid.assign_set_point(self.DEFAULT_SPEED)
        self._steering_history = []
        self._throttle_history = []
        self._car = car
        self._car.register(self)
        self._traffic_sign_classifier = TrafficSignClassifier()


    def on_dashboard(self, src_img, last_steering_angle, speed, throttle, info):
        track_img     = ImageProcessor.preprocess(src_img)
        current_angle = ImageProcessor.find_steering_angle_by_color(track_img, last_steering_angle, debug = self.debug)
        #current_angle = ImageProcessor.find_steering_angle_by_line(track_img, last_steering_angle, debug = self.debug)
        steering_angle = self._steering_pid.update(-current_angle)
        throttle       = self._throttle_pid.update(speed)

        if self.debug:
            ImageProcessor.show_image(src_img, "source")
            ImageProcessor.show_image(track_img, "track")
            #logit("steering PID: %0.2f (%0.2f) => %0.2f (%0.2f)" % (current_angle, ImageProcessor.rad2deg(current_angle), steering_angle, ImageProcessor.rad2deg(steering_angle)))
            #logit("throttle PID: %0.4f => %0.4f" % (speed, throttle))
            #logit("info: %s" % repr(info))

        if self._record_folder:
            suffix = "-deg%0.3f" % (ImageProcessor.rad2deg(steering_angle))
            ImageProcessor.save_image(self._record_folder, src_img  , prefix = "cam", suffix = suffix)
            ImageProcessor.save_image(self._record_folder, track_img, prefix = "trk", suffix = suffix)

        #smooth the control signals
        self._steering_history.append(steering_angle)
        self._steering_history = self._steering_history[-self.MAX_STEERING_HISTORY:]
        self._throttle_history.append(throttle)
        self._throttle_history = self._throttle_history[-self.MAX_THROTTLE_HISTORY:]

        traffic_sign_imgs, _ = ImageProcessor.locate_traffic_signs(src_img)
        if len(traffic_sign_imgs) > 0:
            traffic_signs = self._traffic_sign_classifier.predict(traffic_sign_imgs)
            for i in range(len(traffic_signs)):
                if traffic_signs[i] != 0:
                    print('{} sign detected'.format(traffic_sign_texts[traffic_signs[i]])) 

        self._car.control(sum(self._steering_history)/self.MAX_STEERING_HISTORY, sum(self._throttle_history)/self.MAX_THROTTLE_HISTORY)


class Car(object):
    MAX_STEERING_ANGLE = 40.0


    def __init__(self, control_function):
        self._driver = None
        self._control_function = control_function


    def register(self, driver):
        self._driver = driver


    def on_dashboard(self, dashboard):
        #normalize the units of all parameters
        last_steering_angle = np.pi/2 - float(dashboard["steering_angle"]) / 180.0 * np.pi
        throttle            = float(dashboard["throttle"])
        brake               = float(dashboard["brakes"])
        speed               = float(dashboard["speed"])
        img                 = ImageProcessor.bgr2rgb(np.asarray(Image.open(BytesIO(base64.b64decode(dashboard["image"])))))
        del dashboard["image"]
        #print datetime.now(), dashboard;
        total_time = float(dashboard["time"])
        elapsed    = total_time

        if elapsed >600:
            print "elapsed: " +str(elapsed)
            send_restart()

        info = {
            "lap"    : int(dashboard["lap"]) if "lap" in dashboard else 0,
            "elapsed": elapsed,
            "status" : int(dashboard["status"]) if "status" in dashboard else 0,
        }
        self._driver.on_dashboard(img, last_steering_angle, speed, throttle, info)


    def control(self, steering_angle, throttle):
        #convert the values with proper units
        steering_angle = min(max(ImageProcessor.rad2deg(steering_angle), -Car.MAX_STEERING_ANGLE), Car.MAX_STEERING_ANGLE)
        self._control_function(steering_angle, throttle)


if __name__ == "__main__":
    import shutil
    import argparse
    from datetime import datetime

    import socketio
    import eventlet
    import eventlet.wsgi
    from flask import Flask

    parser = argparse.ArgumentParser(description='AutoDriveBot')
    parser.add_argument(
        'record',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder to record the images.'
    )
    args = parser.parse_args()

    if args.record:
        if not os.path.exists(args.record):
            os.makedirs(args.record)
        logit("Start recording images to %s..." % args.record)

    sio = socketio.Server()
    def send_control(steering_angle, throttle):
        sio.emit(
            "steer",
            data={
                'steering_angle': str(steering_angle),
                'throttle': str(throttle)
            },
            skip_sid=True)
    def send_restart():
        sio.emit(
            "restart",
            data={},
            skip_sid=True)

    car = Car(control_function = send_control)
    drive = AutoDrive(car, args.record)

    @sio.on('telemetry')
    def telemetry(sid, dashboard):
        if dashboard:
            car.on_dashboard(dashboard)
        else:
            sio.emit('manual', data={}, skip_sid=True)

    @sio.on('connect')
    def connect(sid, environ):
        car.control(0, 0)

    app = socketio.Middleware(sio, Flask(__name__))
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

# vim: set sw=4 ts=4 et :
