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

VERSION_NUMBER = 1
AREA_BALL = 70 #210    #150 125 200
CENTER_PIXEL = 319.5   #159.5
CURVE_MIN = 5         # 4
DEFAULT_PARAMETERS_FILENAME = "params.ini"
ROBORIO_SERVER_STATIC_IP = "10.9.10.2"
FONT = cv2.FONT_HERSHEY_SIMPLEX
HORIZONTAL_FOV = 62.2 # degrees, per Pi Camera spec
HORIZONTAL_DEGREES_PER_PIXEL = (HORIZONTAL_FOV / 640) #was 320
VERTICAL_FOV = 48.8 # degrees, per Pi Camera spec
VERTICAL_DEGREES_PER_PIXEL = (VERTICAL_FOV / 480)     #was 240
X_CENTER_ADJUSTMENT = 0
Y_CENTER_ADJUSTMENT = 0
DISTANCE_TO_ROBOT_EDGE = 8.75 #units are inches, measured on comp robot # 7.75 # units are inches, measured on practice robot
VERTICAL_PIXEL_CENTER = 239.5 #was 119.5
NUM_BALLS_TO_FIND = 3 # <= of this number of balls will be reported, based on closest (y pixel value)

# https://urldefense.proofpoint.com/v2/url?u=https-3A__arachnoid.com_polysolve_&d=DwIGAg&c=yzoHOc_ZK-sxl-kfGNSEvlJYanssXN3q-lhj0sp26wE&r=Fp_gwLhy-KcAlNOBY3KaaSSXydGcZGU_kUjMUrqDaFY&m=kzrez1A_s9RwEVeODD_f_fx3Hs2iU10UFhFln-e9LlY&s=z42VPM7SoETsDiNpO4-PDMxolzUXHMAOr-4MBveKSvA&e= 
terms = [
     8.8029826734937001e+002,
    -1.9231902413109701e+001,
     1.6765203801080228e-001,
    -6.5403826962873971e-004,
     9.4477457242682120e-007
]

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

class PiVideoStream: # from pyimagesearch
    def __init__(self, resolution=(640, 480), framerate=40):
        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
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

def check_ball():
    if ( (cv2.getTrackbarPos("ball", "window") == 1)) or (pi_nt.getBoolean("BallEnable",False) is True):
        return True
    else: 
        return False

#cam = PiCamera ()
#cam.resolution = (320, 240)
#cam.framerate = 30
#cam.exposure_mode = "snow"
#cam.awb_mode = "off"
#cam.awb_gains = (1.3, 1.8)
#cam.shutter_speed = 8000
#cam.saturation = -50
#rawcapture = PiRGBArray (cam, size=(320,240))
start_perf = 0
start = 0
debug=0


ball_id = 0
ball_dist_avg = []
ball_angle_avg = []
ball_idx = 0
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
cv2.createTrackbar("delay", "window", 0,1, on_trackbar) # unavailable
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

print("Ball Camera version " + str(VERSION_NUMBER))

NetworkTables.initialize(server=ROBORIO_SERVER_STATIC_IP)
NetworkTables.setUpdateRate(0.010) #was (0.040) - Mr. C 2/28/2021
vis_nt = NetworkTables.getTable("Vision")
pi_nt = NetworkTables.getTable("Pi")

pi_alive_time = time.process_time()
pi_alive_value = 0

# uncomment to capture data to a file
# fo = open("ball_data.csv", "a+")

lower_hue = cv2.getTrackbarPos("ball low h", "window")
lower_saturation = cv2.getTrackbarPos("ball low s", "window")
lower_value = cv2.getTrackbarPos("ball low v", "window")
higher_hue = cv2.getTrackbarPos("ball high h", "window")
higher_saturation = cv2.getTrackbarPos("ball high s", "window")
higher_value = cv2.getTrackbarPos("ball high v", "window")
lower_mask = np.array([lower_hue, lower_saturation, lower_value])
higher_mask = np.array([higher_hue, higher_saturation, higher_value])

vs = PiVideoStream().start() # create camera object and start reading images
time.sleep(3) # camera sensor settling time
fps = FPS()

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

    # Read image array (NumPy format)
    image = vs.read()

    # Convert to hsv for further processing
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if cv2.getTrackbarPos("mode", "window") == 1:
        lower_hue = cv2.getTrackbarPos("ball low h", "window")
        lower_saturation = cv2.getTrackbarPos("ball low s", "window")
        lower_value = cv2.getTrackbarPos("ball low v", "window")
        higher_hue = cv2.getTrackbarPos("ball high h", "window")
        higher_saturation = cv2.getTrackbarPos("ball high s", "window")
        higher_value = cv2.getTrackbarPos("ball high v", "window")
        lower_mask = np.array([lower_hue, lower_saturation, lower_value])
        higher_mask = np.array([higher_hue, higher_saturation, higher_value])
    
    mask = cv2.inRange(hsv, lower_mask, higher_mask)

    # Ball functions
    if (check_ball() == True):

        contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ball_positions = []
        ball_ys = []
        closest_n_ball_positions = []
        closest_n_ball_ys = []
        distance_angle = []
        for i in range(NUM_BALLS_TO_FIND):
            distance_angle.append([0.0,0.0])

        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approxCurve = cv2.approxPolyDP(c, perimeter * 0.02, True)
            area_ball = cv2.contourArea(c)
            (x,y), radius = cv2.minEnclosingCircle(c)
            if  (len (approxCurve) > CURVE_MIN) and (area_ball > AREA_BALL): 
                x += X_CENTER_ADJUSTMENT
                y += Y_CENTER_ADJUSTMENT
                ball_positions.append((x,y,radius))
                ball_ys.append(y)
                center = (int (x), int (y))
                radius = int(radius)

        if (len(ball_ys) > 0):

            while ( (len(closest_n_ball_positions) < NUM_BALLS_TO_FIND) and len(ball_ys) > 0):
                y_max = max(ball_ys)
                closest_n_ball_ys.append(y_max)
                y_max_index = ball_ys.index(y_max)
                closest_n_ball_positions.append(ball_positions[y_max_index])
                ball_ys.pop(y_max_index)
                ball_positions.pop(y_max_index)

            # at this point, there are 2 pair of lists
            # closest_n_ball_ys/closest_n_ball_positions have the y pixel values and x,y,r of <= NUM_BALLS_TO_FIND closest balls, going by y pixel vales
            # ball_ys/ball_positions have the y pixel values and x,y,r of remainging balls in this image, smaller y values then are in closest_n_ball_ys

            # find distance and angle for up to NUM_BALLS_TO_FIND
            for ball in range(len(closest_n_ball_ys)):
                y = closest_n_ball_ys[ball]
                y += Y_CENTER_ADJUSTMENT
                
                x = closest_n_ball_positions[ball][0]
                x += X_CENTER_ADJUSTMENT
                
                radius = closest_n_ball_positions[ball][2]
                
                angle = (int(x) - CENTER_PIXEL) * HORIZONTAL_DEGREES_PER_PIXEL
                real_distance = regress(int(y/2))
                theta2 = math.degrees(math.asin(math.sin(math.radians(math.fabs(angle))*DISTANCE_TO_ROBOT_EDGE / real_distance)))
                theta3 = 180 - (theta2 + math.fabs(angle))
                real_angle_absolute = 180 - theta3
                if(angle < 0):
                    real_angle = -real_angle_absolute
                else:
                    real_angle = real_angle_absolute

                distance_angle[ball][0] = real_distance
                distance_angle[ball][1] = real_angle

            # send distance and angle data to rio
            ball_id += 1
            ball_data = str(ball_id)
            for sr in range(NUM_BALLS_TO_FIND):
                ball_data += (',' + repr(distance_angle[sr][0]) + ',' + repr(distance_angle[sr][1]))
            vis_nt.putString("Ball", ball_data)

            if (cv2.getTrackbarPos("mode", "window") == 1):
               print (ball_data)

            # draw circles on the closest balls - red
            for dr in range(len(closest_n_ball_positions)):
                center = (int(closest_n_ball_positions[dr][0]), int (closest_n_ball_positions[dr][1]))
                radius = closest_n_ball_positions[dr][2]
                cv2.circle(image, center, int(radius), (0,0,255), 2)

            # draw circles on the remaining balls - blue
            for db in range(len(ball_positions)):
                center = (int(ball_positions[db][0]), int (ball_positions[db][1]))
                radius = ball_positions[db][2]
                cv2.circle(image, center, int(radius), (255,0,0), 2)

            '''
            x = ball_positions[y_index][0]
            radius = ball_positions[y_index][2]
            angle = (x - CENTER_PIXEL) * HORIZONTAL_DEGREES_PER_PIXEL
            real_distance = regress(int(y/2))
            theta2 = math.degrees(math.asin(math.sin(math.radians(math.fabs(angle))*DISTANCE_TO_ROBOT_EDGE / real_distance)))
            theta3 = 180 - (theta2 + math.fabs(angle))
            real_angle_absolute = 180 - theta3
            if(angle < 0):
                real_angle = -real_angle_absolute
            else:
                real_angle = real_angle_absolute
            if (len(ball_dist_avg) < 3):
                ball_dist_avg = ball_dist_avg + [real_distance]
                ball_angle_avg = ball_angle_avg + [real_angle]
            else: #(len(ball_dist_avg) == 3 and len(ball_angle_avg) == 3):
                ball_dist_avg[ball_idx] = real_distance
                ball_angle_avg[ball_idx] = real_angle
                ball_idx = ball_idx + 1
                if (ball_idx >= 3):
                    ball_idx = 0

                ball_real_dist_avg = (ball_dist_avg[0] + ball_dist_avg[1] + ball_dist_avg[2])/3
                #ball_dist_avg.clear()
                ball_real_angle_avg = (ball_angle_avg[0] + ball_angle_avg[1] + ball_angle_avg[2])/3
                #ball_angle_avg.clear()
                #ball_data = "%d,%.2f,%.2f,%.2f" % (ball_id, angle, real_distance, real_angle)
                ball_data = "%d,%.2f,%.2f" % (ball_id, ball_real_dist_avg, ball_real_angle_avg)
                ball_id = ball_id + 1
                vis_nt.putString("Ball", ball_data)
                # uncomment the following 2 lines to save data to file
                # csv_ball_data = "%.4f,%s" % (time.process_time(), ball_data)
                # fo.write(csv_ball_data + '\n')
                if (cv2.getTrackbarPos("mode", "window") == 1):
                    print (ball_data)
            '''

            '''
            for i in range(len(ball_ys)):
                center = (int(ball_positions[i][0]), int (ball_positions[i][1]))
                radius = ball_positions[i][2]
                if i == y_index: # if it is the closest ball
                    cv2.circle(image, center, int(radius), (0,0,255), 2) # draw a red circle on the closest ball
                else:
                    cv2.circle(image, center, int(radius), (255,0,0), 2) # draw a blue circle on the other balls
            '''
                    
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
    
    if (debug == 1):
        print("elasped time: {:3.2f}".format(fps.elapsed()*1000))
        #prinf(ball_data)
    
    # Wait for 1ms to see if the escape key was pressed, and if so, exit
    key = cv2.waitKey(1)
    if key == 27: # Press esc, close program
        # uncomment to capture data to a file
        # fo.close()
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
