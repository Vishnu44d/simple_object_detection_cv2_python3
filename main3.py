'''
This code is written by vishnu deo gupta. It is for NSSC challenge. 
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpim
import serial
import bluetooth

#Threshold for the given image..
thres1 = 80
thres2 = 100

#Threshold for the distance before we want to break
#Check the value during competetion.
distance_thresh = 20
#Threshold of angle, it is the most crucial part to set these threshold right
#the current value is set arbitraly, because i don't have arena to test.
angle_thresh1 = 6
angle_thresh2 = 6


## set the path of video here eg video.mp4
FILE_PATH = 0


#Starting communication with ardino
#ser = serial.Serial(port = 'COM3', baudrate = 9600, bytesize = serial.EIGHTBITS, parity = serial.PARITY_NONE, stopbits = serial.STOPBITS_ONE, timeout = 1)

sock = bluetooth.BluetoothSocket( bluetooth.RFCOMM )

#thres1 = 100
#thres2 = 200

def remove_duplicate(l):            #takes the input as a list
    l = list(set(l))
    return l

'''
def approximate(l):                 #approximate the list
    for i in range(l):
        for j in range(l[0]):
    return l
'''
#img = cv2.imread('perfect1.jpg')
def detect_edge(img):
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('.jpeg', processed_img)
    #cv2.imshow('gray_image', processed_img)
    processed_img = cv2.Canny(processed_img, threshold1=thres1, threshold2=thres2)
    processed_img = cv2.GaussianBlur(processed_img,(5, 5), 0)
    return processed_img        

def detect_bot(img):
    processed_img = detect_edge(img)
    #_, thresh = cv2.threshold(processed_img, thresh1,thresh2,0)
    (im2, conts, _) = cv2.findContours(processed_img.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cx = int(M['m10']/M['m00'])
    #cy = int(M['m01']/M['m00'])
    for c in conts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if(len(approx) == 4):
            M = cv2.moments(c)
            print('Bot Found')
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            #print(cx, cy)
    return (cx, cy)

def detect_planet(img):
    processed_img = detect_edge(img)
    #_, thresh = cv2.threshold(processed_img, thresh1,thresh2,0)
    (im2, conts, _) = cv2.findContours(processed_img.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cx = int(M['m10']/M['m00'])
    #cy = int(M['m01']/M['m00'])
    centroids = []
    for c in conts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if(len(approx) == 4):
            pass
        else:
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centroids.append((cx,cy))
        centroids = remove_duplicate(centroids)
        
    print(centroids)
    #print(centroids[0][0])
    return centroids                        #list of tuples of all the centroids...

def euclidian_distance(x1, y1, x2, y2):
    return (np.sqrt((x1-x2)**2 + (y1-y2)**2))

#calculating distances of all the planet from bot and storing in a list which is sorted
def distances(x, y, l):
    distance = []
    for i in l:
        distance.append(euclidian_distance(x,y,i[0],i[1]))
    distance.sort()
    return distance

#center_of_planets = detect_planet(img)
#bot_center_x, bot_center_y = detect_bot(img)

#print(distances(bot_center_x, bot_center_y, center_of_planets))

#visited = [0 for _ in range(8)]
#print(visited)

#(x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
def calc_angle(img):
    processed_img = detect_edge(img)
    #_, thresh = cv2.threshold(processed_img, thresh1,thresh2,0)
    (im2, conts, _) = cv2.findContours(processed_img.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cx = int(M['m10']/M['m00'])
    #cy = int(M['m01']/M['m00'])
    for c in conts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if(len(approx) == 4):
            (x,y),(MA,ma),angle = cv2.fitEllipse(c)
            #print(angle)
            #print(cx, cy)
        #angle = angle * np.pi / 180.0
    return (angle)

def get_angle(img):
    return (calc_angle(img)* np.pi / 180.0)

def desired_angle(x1,y1,x2,y2):
    return np.arctan((y2-y1)/(x2-x1))
    

##def sum_vis(l):
##    s = 0
##    for i in l:
##        s += i
##    return s

def remove_close_value(l):
    d = []
    for i in range(len(l)):
        if i == 0:
            if l[i][0] > l[i+1][0] - 5 and l[i][0] < l[i+1][0] + 5:
                d.append(l[i])
            elif l[i][1] > l[i+1][1] - 5 and l[i][1] < l[i+1][1] + 5:
                d.append(l[i])
        elif i == len(l):
            if l[i][0] > l[i-1][0] - 5 and l[i][0] < l[i-1][0] + 5:
                d.append(l[i])
            elif l[i][1] > l[i-1][1] - 5 and l[i][1] < l[i-1][1] + 5:
                d.append(l[i])
        else:
            if ((l[i][0] > l[i+1][0] - 5 and l[i][0] < l[i+1][0] + 5) or (l[i][0] > l[i-1][0] - 5 and l[i][0] < l[i-1][0] + 5)):
                d.append(l[i])
            elif ((l[i][0] > l[i+1][0] - 5 and l[i][0] < l[i+1][0] + 5) or (l[i][0] > l[i-1][0] - 5 and l[i][0] < l[i-1][0] + 5)):
                d.append(l[i])
            
    return d


#print(get_angle(img))    
##center_of_planets = detect_planet(img)
##bot_center_x, bot_center_y = detect_bot(img)
##distance = distances(bot_center_x, bot_center_y, center_of_planets)
##visited = [0 for _ in range(len(center_of_planets))]


##The main algorithm...

cap = cv2.VideoCapture(FILE_PATH)

i = 0

while(True):
    ret, frame = cap.read()
    #cv2.imshow('video', frame)
    
    #getting centers of planets
    center_of_planets = detect_planet(frame)
    #removing the chance of duplicasy or close value that can come due to faulty edge detection
    center_of_planets = remove_close_value(center_of_planets)
    #creating a list to track the nth planet
    visited = [0 for _ in range(len(center_of_planets))]
    #calculating bot centroid
    bot_center_x, bot_center_y = detect_bot(frame)
    #calculating bot to nearest planet
    distance = distances(bot_center_x, bot_center_y, center_of_planets)
    #calculating bot orientation
    bot_angle = get_angle(frame)
    #calculating bot to planet angle
    bot_to_planet_angle = desired_angle(bot_center_x, bot_center_y, center_of_planets[i][0], center_of_planets[i][1])

    
    if sum(visited) == len(center_of_planets):
        break
    else:
        while(distance[i] > distance_thresh):
            if (bot_angle > bot_to_planet_angle + angle_thresh1):
                #give command to turn left
                sock.send('l')
            elif (bot_angle < bot_to_planet_angle - angle_thresh2):
                #give command to turn right
                sock.send('r')
            else:
                sock.send('f')
                #give command to go straight
            
        visited[i] = 1
        sock.send(i)
        i += 1
        cv2.imshow('At Arena', frame)
        #sock.send(i)
        #pass the value of i in arduino

sock.close()
cv2.waitKey(0)
cv2.destroyAllWindows()

#end of the code


'''
for i in range(len(distance)):
            while(distance[i]>distance_thresh):
                if ((get_angle(img))>left_angle_low_thresh and get_angle(img) < left_angle_high_thresh):
                    #give turn left command
                else if ((get_angle(img))>right_angle_low_thresh and get_angle(img) < right_angle_high_thresh):
                    #give turn right command
                #give forward command
            else:
                #give stop command
                #give the value of i to arduino


#print(bot_center_x, bot_center_y)
#processed_img = detect_edge(img)

#cv2.imshow('image', processed_img)
#cv2.imwrite('processed.jpeg',processed_img)

#plt.figure()
#plt.imshow(gray_image)
#plt.show()
'''
