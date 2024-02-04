######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import pdb
import time
import math
import pathlib
from threading import Thread
import importlib.util
import datetime
import tflite_runtime.interpreter as tflite


import time

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        #breakpoint()
        
        self.stream = cv2.VideoCapture(0)
        print("Camera initiated.")
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--classmoddir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected keypoints (specify between 0 and 1).',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
parser.add_argument('--output_path', help="Where to save processed imges from pi.",
                    required=True)

args = parser.parse_args()

MODEL_NAME = args.modeldir
model_path = args.classmoddir
#MODEL1 = args.model1dir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu
output_path = args.output_path

# Import TensorFlow libraries
# If tensorflow is not installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = None
#pkg = importlib.util.find_spec('tensorflow')
if pkg is None:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME)


# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(output_details)
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
#set stride to 32 based on model size
OUTPUT_STRIDE = 4

led_on = False
floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

def mod(a, b):
    """find a % b"""
    floored = np.floor_divide(a, b)
    return np.subtract(a, np.multiply(floored, b))

def sigmoid(x):
    """apply sigmoid actiation to numpy array"""
    return 1/ (1 + np.exp(-x))
    
def sigmoid_and_argmax2d(inputs, threshold):
    """return y,x coordinates from heatmap"""
    #v1 is 9x9x17 heatmap
    v1 = interpreter.get_tensor(output_details[0]['index'])[0]
    height = v1.shape[0]
    width = v1.shape[1]
    depth = v1.shape[2]
    reshaped = np.reshape(v1, [height * width, depth])
    reshaped = sigmoid(reshaped)
    #apply threshold
    reshaped = (reshaped > threshold) * reshaped
    coords = np.argmax(reshaped, axis=0)
    yCoords = np.round(np.expand_dims(np.divide(coords, width), 1)) 
    xCoords = np.expand_dims(mod(coords, width), 1) 
    return np.concatenate([yCoords, xCoords], 1)

def get_offset_point(y, x, offsets, keypoint, num_key_points):
    """get offset vector from coordinate"""
    y_off = offsets[y,x, keypoint]
    x_off = offsets[y,x, keypoint+num_key_points]
    return np.array([y_off, x_off])

def post_process_heatmap_simple(heatmap, conf_threshold=1e-6):
    """
    A simple approach of keypoints heatmap post process,
    only pick 1 max point in each heatmap as keypoint output
    """
    keypoint_list = list()
    for i in range(heatmap.shape[-1]):
        # ignore last channel, background channel
        _map = heatmap[:, :, i]
        # clear value less than conf_threshold
        under_th_indices = _map < conf_threshold
        _map[under_th_indices] = 0

        # choose the max point in heatmap (we only pick 1 keypoint in each heatmap)
        # and get its coordinate & confidence
        y, x = np.where(_map == _map.max())
        if len(x) > 0 and len(y) > 0:
            keypoint_list.append((int(x[0]), int(y[0]), _map[y[0], x[0]]))
        else:
            keypoint_list.append((0, 0, 0))
    return keypoint_list

def get_offsets(output_details, coords, num_key_points=17):
    """get offset vectors from all coordinates"""
    offsets = interpreter.get_tensor(output_details[1]['index'])
    print(offsets)
    offset_vectors = np.array([]).reshape(-1,2)
    for i in range(len(coords)):
        heatmap_y = int(coords[i][0])
        heatmap_x = int(coords[i][1])
        print(heatmap_x)
        #make sure indices aren't out of range
        if heatmap_y >8:
            heatmap_y = heatmap_y -1
        if heatmap_x > 8:
            heatmap_x = heatmap_x -1
        offset_vectors = np.vstack((offset_vectors, get_offset_point(heatmap_y, heatmap_x, offsets, i, num_key_points)))  
    return offset_vectors

def draw_lines(keypoints, image, bad_pts):
    """connect important body part keypoints with lines"""
    #color = (255, 0, 0)
    color = (0, 255, 0)
    thickness = 2
    #refernce for keypoint indexing: https://www.tensorflow.org/lite/models/pose_estimation/overview
    body_map = [[5,6], [5,7], [7,9], [5,11], [6,8], [8,10], [6,12], [11,12], [11,13], [13,15], [12,14], [14,16]]
    for map_pair in body_map:
        #print(f'Map pair {map_pair}')
        if map_pair[0] in bad_pts or map_pair[1] in bad_pts:
            continue
        start_pos = (int(keypoints[map_pair[0]][1]), int(keypoints[map_pair[0]][0]))
        end_pos = (int(keypoints[map_pair[1]][1]), int(keypoints[map_pair[1]][0]))
        image = cv2.line(image, start_pos, end_pos, color, thickness)
    return image

def process_heatmap(heatmap, image_file, image, scale, class_names, skeleton_lines, output_path):
    start = time.time()
    # parse out predicted keypoint from heatmap
    keypoints = post_process_heatmap_simple(heatmap)

    # rescale keypoints back to origin image size
    keypoints_dict = dict()
    for i, keypoint in enumerate(keypoints):
        keypoints_dict[class_names[i]] = (keypoint[0] * scale[0] * OUTPUT_STRIDE, keypoint[1] * scale[1] * OUTPUT_STRIDE, keypoint[2])

    end = time.time()
    print("PostProcess time: {:.8f}ms".format((end - start) * 1000))

    #print('Keypoints detection result:')
    #for keypoint in keypoints_dict.items():
        #print(keypoint)

    # draw the keypoint skeleton on image
    image_array = np.array(image, dtype='uint8')
    image_array = render_skeleton(image_array, keypoints_dict, skeleton_lines)

    # save or show result
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, os.path.basename(image_file))
        # Image.fromarray(image_array).save(output_file)
        cv2.imwrite( output_file, image_array)
    # else:
    #     Image.fromarray(image_array).show()
    return image_array
    
def normalize_image(imgdata, color_mean):
    '''
    :param imgdata: image in 0 ~ 255
    :return:  image from 0.0 to 1.0
    '''
    imgdata = imgdata / 255.0

    for i in range(imgdata.shape[-1]):
        imgdata[:, :, i] -= color_mean[i]

    return imgdata
    
def preprocess_image(image, model_input_shape, mean=(0.4404, 0.4440, 0.4327)):
    """
    Prepare model input image data with
    resize, normalize and dim expansion

    # Arguments
        image: origin input image
            PIL Image object containing image data
        model_input_shape: model input image shape
            tuple of format (height, width).

    # Returns
        image_data: numpy array of image data for model input.
    """
    # resized_image = image.resize(tuple(reversed(model_input_shape)), Image.BICUBIC)
    height, width = model_input_shape
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    image_data = np.asarray(resized_image).astype('float32')

    mean = np.array(mean, dtype=float)
    image_data = normalize_image(image_data, mean)
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension
    return image_data

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_skeleton(skeleton_path):
    '''loads the skeleton'''
    with open(skeleton_path) as f:
        skeleton_lines = f.readlines()
    skeleton_lines = [s.strip() for s in skeleton_lines]
    return skeleton_lines

def render_skeleton(image, keypoints_dict, skeleton_lines=None, conf_threshold=0.001, colors=None):
    """
    Render keypoints skeleton on provided image with
    keypoints dict and skeleton lines definition.
    If no skeleton_lines provided, we'll only render
    keypoints.
    """
    def get_color(color_pattern):
        color = (255, 0, 0)

        if color_pattern == 'r':
            color = (255, 0, 0)
        elif color_pattern == 'g':
            color = (0, 255, 0)
        elif color_pattern == 'b':
            color = (0, 0, 255)
        else:
            raise ValueError('invalid color pattern')

        return color

    def draw_line(img, start_point, end_point, color=(255, 0, 0)):
        x_start, y_start, conf_start = start_point
        x_end, y_end, conf_end = end_point

        if (x_start > 1 and y_start > 1 and conf_start > conf_threshold) and (x_end > 1 and y_end > 1 and conf_end > conf_threshold):
            cv2.circle(img, center=(int(x_start), int(y_start)), color=color, radius=3, thickness=-1)
            cv2.circle(img, center=(int(x_end), int(y_end)), color=color, radius=3, thickness=-1)
            cv2.line(img, (int(x_start), int(y_start)), (int(x_end), int(y_end)), color=color, thickness=1)
        return img

    def draw_keypoints(img, key_points, color):
        for key_point in key_points:
            x, y, conf = key_point
            if x > 1 and y > 1 and conf > conf_threshold:
                cv2.circle(img, center=(int(x), int(y)), color=color, radius=3, thickness=-1)
        return img

    if skeleton_lines:
        for skeleton_line in skeleton_lines:
            #skeleton line format: [start_point_name,end_point_name,color]
            skeleton_list = skeleton_line.split(',')
            color = colors
            if color is None:
                color = get_color(skeleton_list[2])
            image = draw_line(image, keypoints_dict[skeleton_list[0]], keypoints_dict[skeleton_list[1]], color=color)
    else:
        if colors is None:
            colors = (0, 0, 0)
        image = draw_keypoints(image, list(keypoints_dict.values()), colors)

    return image
    
def post_processkp( keypoints, predicted_class):
    # print(  eval(keypoints['right_ankle'][0]))
    l_w  = eval(keypoints['left_wrist'][0])[:2]
    r_w = eval(keypoints['right_wrist'][0])[:2]
    head = eval(keypoints['thorax'][0])[:2]    
    l_k = eval(keypoints['left_knee'][0])[:2]
    r_k = eval(keypoints['right_knee'][0])[:2]
    l_a = eval(keypoints['left_ankle'][0])[:2]
    r_a = eval(keypoints['right_ankle'][0])[:2]

    print( l_w, r_w)

    if predicted_class == 0: # lateral raise 
        print( "Logic for lateral raise")
        # the wrist should not be higher than head 
        if l_w[1] > head[1] or r_w[1] > head[1]:        
            return "Bad form"
        else:
            return "Good Form"
    elif predicted_class == 1: # pushups 
        # the wrist should not be at the same head level 
        if l_w[1] > head[1] or r_w[1] > head[1]:        
            return "Bad form"
        else:
            return "Good Form"
    else:# squats 
        # either knee should be greater than the foot 
        if l_k[0] > l_a[0] or r_k[0] < r_a[0]:
            return "Bad form"
        else:
            return "Good Form"

interpreter = tflite.Interpreter(MODEL_NAME)
interpreter.allocate_tensors()

interpreter1 = tflite.Interpreter(model_path)
interpreter1.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
model_input_shape = (height, width)

classes_path = '/home/pi/rpi_pose_estimation/mpii_classes.txt'
skeleton_path = '/home/pi/rpi_pose_estimation/mpii_skeleton.txt'


#flag for debugging
debug = True 

try:
    print("Progam started - waiting for button push...")
    while True:
        #make sure LED is off and wait for button press
        if True:
            #timestamp an output directory for each capture
            outdir = pathlib.Path(args.output_path) / time.strftime('%Y-%m-%d_%H-%M-%S-%Z')
            outdir.mkdir(parents=True)
            time.sleep(.1)
            led_on = True
            f = []

            # Initialize frame rate calculation
            frame_rate_calc = 1
            freq = cv2.getTickFrequency()
            videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
            time.sleep(1)

            #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
            while True:
                print('running loop')
                # Start timer (for calculating frame rate)
                t1 = cv2.getTickCount()
                # Grab frame from video stream
                frame1 = videostream.read()
                image_file = "/home/pi/rpi_pose_estimation/01.jpg"
                # Acquire frame and resize to expected shape [1xHxWx3]
                frame = frame1.copy()
                image_data = preprocess_image(frame, model_input_shape)
                image_size = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (width, height))
                input_data = np.expand_dims(frame_resized, axis=0)
                frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                scale = (image_size[1] * 1.0 / model_input_shape[1], image_size[0] * 1.0 / model_input_shape[0])

                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()

                prediction = []
                for output_detail in output_details:
                    output_data = interpreter.get_tensor(output_detail['index'])
                    prediction.append(output_data)

                heatmap = prediction[-1][0]
                class_names = get_classes(classes_path)
                skeleton_lines = get_skeleton(skeleton_path)    
                image_array = process_heatmap(heatmap, image_file, frame, scale, class_names, skeleton_lines, output_path)
                
                interpreter1.set_tensor(input_details[0]['index'], image_array)
                interpreter1.invoke()
                
                utput_data = interpreter.get_tensor(output_details[0]['index'])
                predicted_class = np.argmax(output_data)
                cv2.imshow('image', image_array)
                cv2.waitKey(1)

except KeyboardInterrupt:
    cv2.destroyAllWindows()
    videostream.stop()
    print('Stopped video stream.')
    GPIO.output(4, False)
    GPIO.cleanup()
