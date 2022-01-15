#TEMP 2/25
import cv2
from picamera.array import PiRGBArray 
from picamera import PiCamera
import time
import numpy as np
import PySimpleGUI as sg
import configparser
from networktables import NetworkTables
import math
from threading import Thread

CURVE_MIN = 4
DEFAULT_PARAMETERS_FILENAME = "params.ini"
ROBORIO_SERVER_STATIC_IP = "10.9.10.2"
TARGET_MAX_RATIO = 18
TARGET_MIN_RATIO = 6
FONT = cv2.FONT_HERSHEY_SIMPLEX
HORIZONTAL_FOV = 62.2 # degrees, per Pi Camera spec
HORIZONTAL_DEGREES_PER_PIXEL = (HORIZONTAL_FOV / 640)
VERTICAL_FOV = 48.8 # degrees, per Pi Camera spec
VERTICAL_FOV_MEASURED = 24.13 * 2 # Measured distance between 2 dots on whiteboard at outside edges of view (43 inches) at 48 inches away
VERTICAL_DEGREES_PER_PIXEL = (VERTICAL_FOV / 480)
CAMERA_HEIGHT = 27.25 # inches, from floor
TARGET_HEIGHT = 81 # units are 1nches, location : center of the bottom edge of tape to floor
CAMERA_MOUNT_ANGLE = 24.8+14 # change it to (24.8+14) degrees when the camera mount is changed
X_CENTER_ADJUSTMENT = 0
Y_CENTER_ADJUSTMENT = 0
DISTANCE_TO_ROBOT_EDGE = 7.75 # units are inches, measured on practice robot
VERTICAL_PIXEL_CENTER = 239.5
HORIZONTAL_PIXEL_CENTER = 319.5
TARGET_CAMERA_MOUNT_OFFSET = 3.125 # units are inches, measured on practice robot
DISTANCE_Y = 30.625 #distance from camera to edge of robot in y dimension
X_OFFSET = 0
Y_OFFSET = 0
VERTICAL_TARGET_LENGTH = 39.25
HORIZONTAL_TARGET_LENGTH = 17
TARGET_RATIO = (VERTICAL_TARGET_LENGTH / HORIZONTAL_TARGET_LENGTH)


#New slope based on correct ordering of: Pixels then Inches and NO reverse 4th order polynomial
terms = [
     2.7046874736524629e+001,
    -1.0160419221361943e+000,
     1.9129057536948216e-002,
    -1.1581593345974605e-004,
     3.0019220483095032e-007
]

'''
terms = [
     5.6040571527843596e+001,
    -2.1425054924564124e+000,
     3.4841991969106360e-002,
    -2.0066305633307399e-004,
     4.5114432277218616e-007
]
'''

def regress(x):
  t = 1
  r = 0
  for c in terms:
    r += c * t
    t *= x
  return r

def on_trackbar():
    pass 

def save_parameters(filename):
    print("Saving " + filename)
    parser = configparser.ConfigParser()
    parser.add_section('Parameters')
    parser.set('Parameters', 'ball', str(cv2.getTrackbarPos("ball", "window")))
    parser.set('Parameters', 'delay', str(cv2.getTrackbarPos("delay", "window")))
    parser.set('Parameters', 'network table', str(cv2.getTrackbarPos("network table", "window")))
    parser.set('Parameters', 'ball low h', str(cv2.getTrackbarPos("ball low h", "window")))
    parser.set('Parameters', 'ball low s', str(cv2.getTrackbarPos("ball low s", "window")))
    parser.set('Parameters', 'ball low v', str(cv2.getTrackbarPos("ball low v", "window")))
    parser.set('Parameters', 'ball high h', str(cv2.getTrackbarPos("ball high h", "window")))
    parser.set('Parameters', 'ball high s', str(cv2.getTrackbarPos("ball high s", "window")))
    parser.set('Parameters', 'ball high v', str(cv2.getTrackbarPos("ball high v", "window")))
    parser.set('Parameters', 'target', str(cv2.getTrackbarPos("target", "window")))
    parser.set('Parameters', 'target low h', str(cv2.getTrackbarPos("target low h", "window")))
    parser.set('Parameters', 'target low s', str(cv2.getTrackbarPos("target low s", "window")))
    parser.set('Parameters', 'target low v', str(cv2.getTrackbarPos("target low v", "window")))
    parser.set('Parameters', 'target high h', str(cv2.getTrackbarPos("target high h", "window")))
    parser.set('Parameters', 'target high s', str(cv2.getTrackbarPos("target high s", "window")))
    parser.set('Parameters', 'target high v', str(cv2.getTrackbarPos("target high v", "window")))
   
    # Add target parameters
    fp=open(filename,'w')
    parser.write(fp)
    fp.close()
    print("Saved " + filename)

def load_parameters(filename):
    print("Loading " + filename)
    parser = configparser.ConfigParser()
    parser.read(filename)
    for sect in parser.sections():
        for k,v in parser.items(sect):
            cv2.setTrackbarPos(k, "window", int(v))
    print("Loaded " + filename)
    
def check_target():
    if ( (cv2.getTrackbarPos("target", "window") == 1)) or (pi_nt.getBoolean("TgtEnable",False) is True):
        return True
    else: 
        return False

class PiVideoStream: # from pyimagesearch
	def __init__(self, resolution=(640, 480), framerate=40):
		# initialize the camera and stream
		self.camera = PiCamera()
		self.camera.resolution = resolution
		self.camera.framerate = framerate
		self.camera.exposure_mode = "snow"
		self.camera.awb_mode = "off"
		self.camera.awb_gains = (1.3, 1.8)
		self.camera.shutter_speed = 8000
		self.camera.saturation = -50
		self.rawCapture = PiRGBArray(self.camera, size=resolution)
		self.stream = self.camera.capture_continuous(self.rawCapture,
			format="bgr", use_video_port=True)
		# initialize the frame and the variable used to indicate
		# if the thread should be stopped
		self.frame = None
		self.stopped = False
	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self
	def update(self):
		# keep looping infinitely until the thread is stopped
		for f in self.stream:
			# grab the frame from the stream and clear the stream in
			# preparation for the next frame
			self.frame = f.array
			self.rawCapture.truncate(0)
			# if the thread indicator variable is set, stop the thread
			# and release camera resources
			if self.stopped:
				self.stream.close()
				self.rawCapture.close()
				self.camera.close()
				return
	def read(self):
		# return the frame most recently read
		return self.frame
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

class FPS: # outline from pyimagesearch 
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self.end = None
		self.numFrames = 0
	def start(self):
		# start the timer
		self._start = time.perf_counter()
		return self
	def stop(self):
		# stop the timer
		self.end = time.perf_counter()
	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self.numFrames += 1
	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self.end - self._start)

target_id = 0

lower_hue= 0
lower_saturation = 0
lower_value = 0
higher_hue = 0
higher_saturation= 0
higher_value = 0
# Setup window for saving parameters
layout = [[sg.Text('use as default')],            
                 [sg.Submit(), sg.Cancel()]] 
sg.theme('DarkBlue1')
cv2.namedWindow("window")
cv2.createTrackbar("mode", "window" , 0, 1, on_trackbar)
cv2.createTrackbar("ball", "window", 0, 1, on_trackbar)
cv2.createTrackbar("delay", "window", 0,1, on_trackbar)
cv2.createTrackbar("network table", "window", 0,1, on_trackbar)
cv2.createTrackbar("ball low h", "window" , 0, 180, on_trackbar)
cv2.createTrackbar("ball low s", "window" , 0, 255, on_trackbar)
cv2.createTrackbar("ball low v", "window" , 0, 255, on_trackbar)
cv2.createTrackbar("ball high h", "window" , 0, 180, on_trackbar)
cv2.createTrackbar("ball high s", "window" , 0, 255, on_trackbar)
cv2.createTrackbar("ball high v", "window" , 0, 255, on_trackbar)
cv2.createTrackbar("target", "window", 0, 1, on_trackbar)
cv2.createTrackbar("target low h", "window", 0, 180, on_trackbar)
cv2.createTrackbar("target low s", "window", 0, 255, on_trackbar)
cv2.createTrackbar("target low v", "window", 0, 255, on_trackbar)
cv2.createTrackbar("target high h", "window" , 0, 180, on_trackbar)
cv2.createTrackbar("target high s", "window" , 0, 255, on_trackbar)
cv2.createTrackbar("target high v", "window" , 0, 255, on_trackbar)

load_parameters(DEFAULT_PARAMETERS_FILENAME)

if(cv2.getTrackbarPos("delay", "window") == 1):
    time.sleep(16)

NetworkTables.initialize(server=ROBORIO_SERVER_STATIC_IP)
NetworkTables.setUpdateRate(0.040)
vis_nt = NetworkTables.getTable("Vision")
pi_nt = NetworkTables.getTable("Pi")

pi_alive_time = time.process_time()
pi_alive_value = 0

# uncomment to capture data to file
# fo = open("data_captured.csv", "a+")

vs = PiVideoStream().start() # create camera object and start reading images
time.sleep(3) # camera sensor settling time
fps = FPS()

lower_hue = cv2.getTrackbarPos("target low h", "window")
lower_saturation = cv2.getTrackbarPos("target low s", "window")
lower_value = cv2.getTrackbarPos("target low v", "window")
higher_hue = cv2.getTrackbarPos("target high h", "window")
higher_saturation = cv2.getTrackbarPos("target high s", "window")
higher_value = cv2.getTrackbarPos("target high v", "window")
lower_mask = np.array([lower_hue, lower_saturation, lower_value])
higher_mask = np.array([higher_hue, higher_saturation, higher_value])

while True:

    fps.start()
    
    start = time.process_time()

    if ((start - pi_alive_time) > 1.0):
        pi_alive_value = pi_alive_value + 1
        pi_nt.putValue("pi_alive",pi_alive_value)
        pi_alive_time = time.process_time()

        if (not NetworkTables.isConnected()):
            NetworkTables.initialize(server=ROBORIO_SERVER_STATIC_IP)
            NetworkTables.setUpdateRate(0.040)
            vis_nt = NetworkTables.getTable("Vision")
            pi_nt = NetworkTables.getTable("Pi")

    if (check_target() == True):
    
        # Read image array (NumPy format)
        image = vs.read()
        # rotate the image 180 degrees because the camera is mounted upside down
        image = cv2.rotate(image, cv2.ROTATE_180)

        # Convert to hsv for further processing
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
        if cv2.getTrackbarPos("mode", "window") == 1:
            lower_hue = cv2.getTrackbarPos("target low h", "window")
            lower_saturation = cv2.getTrackbarPos("target low s", "window")
            lower_value = cv2.getTrackbarPos("target low v", "window")
            higher_hue = cv2.getTrackbarPos("target high h", "window")
            higher_saturation = cv2.getTrackbarPos("target high s", "window")
            higher_value = cv2.getTrackbarPos("target high v", "window")
            lower_mask = np.array([lower_hue, lower_saturation, lower_value])
            higher_mask = np.array([higher_hue, higher_saturation, higher_value])

        mask = cv2.inRange(hsv, lower_mask, higher_mask)
                    
        # Target functions
    
        # Find the target by calculating a ratio based on the area of a contour ()> 100) to the area of bounding rectangle for that contour and see if that ratio is in a certain range
        contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)

            area_rect = w*h
            if (area_rect > 100):
                area_target = cv2.contourArea(c)
                target_ratio = (area_target/area_rect)*100
                
                if (target_ratio < TARGET_MAX_RATIO and target_ratio > TARGET_MIN_RATIO):
                    approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)


                    print ("raw x=" + "{:3.2f}".format(x) + ",raw y=" + "{:3.2f}".format(y) )
                                # https://arachnoid.com/polysolve/index.html
                    # camera angle = 24.8 + 14
                    # camera height = 27.25

                    y_corrected = (y+h) / 2

                    target_distance = regress((y_corrected+Y_OFFSET))

                    angle_of = (((x + X_OFFSET) + w/2)- HORIZONTAL_PIXEL_CENTER) * HORIZONTAL_DEGREES_PER_PIXEL

                    # angle_to code (in progress):
                    # Used to flatted the array containing 
                    # the co-ordinates of the vertices. 
                    """
                    n = approx.ravel()
                    #print("len(n)=" + str(len(n)))
                    i = 0
                    x1_min = 999
                    x2_max = 0
                    y1_min = 0
                    y2_max = 0
                    for j in n : 
                        if(i % 2 == 0): 
                            x_current = n[i] 
                            y_current = n[i + 1] 
    
                            if(x_current < x1_min):
                                x1_min = x_current
                                y1_min = y_current

                            if(x_current > x2_max):
                                x2_max = x_current
                                y2_max = y_current

                        i = i + 1
                    """

                    angle_to = 0

                    # get time stamp to determine latency for roborio
                    latency_time_stamp = pi_nt.getValue("Timer",0)
                    target_data = "%.3f,%.2f,%.2f,%.2f" % (latency_time_stamp, target_distance, angle_of, angle_to)
                    target_id = target_id + 1
                    
                    vis_nt.putString("Target", target_data)
                    if (cv2.getTrackbarPos("mode", "window") == 1):
                        #print ("x=" + "{:3.2f}".format(x) + ",bot_y=" + "{:3.2f}".format(y+h) + "tar_a=" + "{:3.2f}".format(area_target) + ",tar_rect=" + "{:3.2f}".format(area_rect) +",tar_ratio = " + "{:3.2f}".format(target_ratio) )
                        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
                        cv2.drawContours(image, [approx], 0, (0, 0, 255), 2)
                        print (target_data)
                        # print(target_id)

                        # debug for angle_to
                        #string_min = str(x1_min) + " " + str(y1_min)  
                        #center = (int(x1_min),int(y1_min))
                        #cv2.circle(image,center,5,(0,255,0),1)
                        #string_max = str(x2_max) + " " + str(y2_max)  
                        #center = (int(x2_max),int(y2_max))
                        #cv2.circle(image,center,5,(255,255,0),1)
                        #slope = (y2_max - y1_min) / (x2_max - x1_min)
                        #print("x1_min,y1_min=" + string_min + " x2_max,y2_max=" + string_max + " slope=" + "{:3.3f}".format(slope))

                    break
                        
        # Display images
        debug = cv2.getTrackbarPos("mode", "window")
        if (debug == 1):
            cv2.imshow("Image", image)
            cv2.imshow("mask", mask)
        else:   
            if (cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) == 1):
                cv2.destroyWindow("Image")
            if (cv2.getWindowProperty("mask", cv2.WND_PROP_VISIBLE) == 1):
                cv2.destroyWindow("mask")

    fps.stop()
    
    #if (cv2.getTrackbarPos("mode", "window") == 1):
    #    print("dt:" + "{:3.2f}".format(fps.elapsed()*1000))

    # Wait for 1ms to see if the escape key was pressed, and if so, exit
    key = cv2.waitKey(1)
    
    if key == 27: # Press esc, close program
        # uncomment to capture data to a file
        #fo.close()
        break
    elif key == ord('s'): # Press 's', save parameters
        fname = sg.popup_get_file('Save Parameters')
        if not fname:
            sg.popup("Warning", "No filename supplied, using " + DEFAULT_PARAMETERS_FILENAME)
            fname = DEFAULT_PARAMETERS_FILENAME
        else:
            sg.popup('The filename you chose was', fname)
        save_parameters(fname) # fname has the filename

cv2.destroyAllWindows()
vs.stop()