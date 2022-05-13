import cv2, requests
import numpy as np
import dlib
from math import hypot


#url = 'http://192.168.100.233:8080'
cap = cv2.VideoCapture(0)  #0 znaci mojata webcamera so e edna

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #68 posebni tocki na liceto(landmarks)

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

# def average(a_list):  # returns average value of a list of numbers.
#     total = 0
#     for n in a_list:
#         total += n
#     return total / len(a_list)
# def count_img_pixels(img,width,height):
#     x_pos = 0
#     y_pos = 1
#     counter = 0
#     for item in img:
#         if (x_pos) == width:
#             x_pos = 1
#             y_pos += 1
#         else:
#             x_pos += 1
        
#         if item[3] != 0:
#             counter = counter+1
#     return counter

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio


#function for gaze detection
def get_gaze_ratio(which_eye,eye_points, facial_landmarks):
    eye_region = np.array([ (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                            (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                            (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                            (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                            (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                            (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)
    
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask,  [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask) #deka za dvete oci se rabotit
    

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
    
    gray_eye = eye[min_y: max_y, min_x: max_x]

    _, threshold_eye = cv2.threshold(gray_eye, 127, 255, cv2.THRESH_BINARY)
    #print(type(threshold_eye))
    height, width = threshold_eye.shape # ja zema slikata od okoto vo crnobela negovata visina i shirina
    
    left_side_of_the_eye_threshold = threshold_eye[0: height, 0: int(width / 2)]# ja vagja levata polovina od okoto za vo nareden chekor da bara dali e bela povekje ili zenica crna
    #left_side_of_the_eye_white_pixels = cv2.countNonZero(left_side_of_the_eye_threshold)
    left_side_of_the_eye_white_pixels = np.sum(left_side_of_the_eye_threshold == 255)
    left_side_of_the_eye_black_pixels = np.sum(left_side_of_the_eye_threshold == 0)
    left_side_of_the_eye_all_pixels = left_side_of_the_eye_white_pixels+left_side_of_the_eye_black_pixels


    right_side_of_the_eye_threshold = threshold_eye[0: height, int(width / 2): width]# ja vagja desnata polovina od okoto za vo nareden chekor da bara dali e bela povekje ili zenica crna
    right_side_of_the_eye_white_pixels = np.sum(right_side_of_the_eye_threshold == 255)
    right_side_of_the_eye_black_pixels = np.sum(right_side_of_the_eye_threshold == 0)
    right_side_of_the_eye_all_pixels = right_side_of_the_eye_white_pixels+right_side_of_the_eye_black_pixels
    
    # countNonZero() to count the number of pixels that are not black (>0) in an image
    # 0 znachi deka site pikseli se crni
    kol_right_look=(right_side_of_the_eye_white_pixels/right_side_of_the_eye_all_pixels)*100
    kol_left_look=(left_side_of_the_eye_white_pixels/left_side_of_the_eye_all_pixels)*100
    #print('left_right % left  look   '+str(kol_left_look))
    #print('left_right % right look   '+str(kol_right_look))
    
    if (abs(kol_right_look - kol_left_look) < 10):
        gaze_ratio_left_right = 1
    elif kol_right_look > kol_left_look:
        gaze_ratio_left_right = 2
    elif kol_left_look > kol_right_look:
        gaze_ratio_left_right = 0
    #print(str(gaze_ratio_left_right))  
        
    up_side_of_the_eye_threshold = threshold_eye[0: width, int(height / 2): height ]# ja vagja levata polovina od okoto za vo nareden chekor da bara dali e bela povekje ili zenica crna
    up_side_of_the_eye_white_pixels = np.sum(up_side_of_the_eye_threshold == 255)
    up_side_of_the_eye_black_pixels = np.sum(up_side_of_the_eye_threshold == 0)
    up_side_of_the_eye_all_pixels = up_side_of_the_eye_white_pixels+up_side_of_the_eye_black_pixels
    
    down_side_of_the_eye_threshold = threshold_eye[0: width, 0: int(height / 2)]# ja vagja levata polovina od okoto za vo nareden chekor da bara dali e bela povekje ili zenica crna
    #left_side_of_the_eye_white_pixels = cv2.countNonZero(left_side_of_the_eye_threshold)
    down_side_of_the_eye_white_pixels = np.sum(down_side_of_the_eye_threshold == 255)
    down_side_of_the_eye_black_pixels = np.sum(down_side_of_the_eye_threshold == 0)
    down_side_of_the_eye_all_pixels = down_side_of_the_eye_white_pixels+down_side_of_the_eye_black_pixels
    
    kol_up_look=(up_side_of_the_eye_black_pixels/up_side_of_the_eye_all_pixels)*100
    kol_down_look=(down_side_of_the_eye_black_pixels/down_side_of_the_eye_all_pixels)*100
    #print('up_down % up   look   '+str(kol_up_look))
    #print('up_down % down look   '+str(kol_down_look))
    
    if (abs(kol_up_look - kol_down_look) < 7):
        gaze_ratio_up_down = 1
    elif kol_up_look > kol_down_look:
        gaze_ratio_up_down = 2
    elif kol_down_look > kol_up_look:
        gaze_ratio_up_down = 0
    #print(str(gaze_ratio_up_down))  
   
    #print('left_right'+str(gaze_ratio_left_right))
    #print('down_up'+str(gaze_ratio_down_up))
    return gaze_ratio_left_right, gaze_ratio_up_down  #go vrakja gaze ratio

list_gaze_avg_value = [0]*10
list_iter_pos=0
while True: 
    _, frame = cap.read()  #citanje na frejmovi
    new_frame = np.zeros((500, 500, 3), np.uint8)  #ovde gi definirame i novata frame i site frajmovi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray) #obejctot detector od prethodno, faces ke e niza od detektiranite lica
    #se dobivaat kako koordinati na liceto sto se detektira od kamerata, top left and right bottom se dvete koordinati
    
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)  #predictor e object za da gi najdit landmarks

        # Detect blinking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks) #ovde go zemat ratioto na dvete oci
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        

        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))


        # Gaze detection
        gaze_ratio_left_eye_left_right, gaze_ratio_left_eye_down_up = get_gaze_ratio('left_eye',[36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye_left_right, gaze_ratio_right_eye_down_up = get_gaze_ratio('right_eye',[42, 43, 44, 45, 46, 47], landmarks)
   
        gaze_ratio_lr = (gaze_ratio_right_eye_left_right + gaze_ratio_left_eye_left_right) / 2 # sredna vrednost od gaze vrednosta na dvete oci kaj gledaat
        #print('left_eye' +'   '+str(gaze_ratio_left_eye_left_right))
        #print('right_eye'+'   '+str(gaze_ratio_right_eye_left_right))
        gaze_ratio_du = (gaze_ratio_right_eye_down_up + gaze_ratio_left_eye_down_up) / 2
        #list_gaze_avg_value[list_iter_pos] = gaze_ratio
        #gaze_ratio = average(list_gaze_avg_value)
        cv2.putText(frame, str(gaze_ratio_lr), (50, 100), font, 2, (0, 0, 255), 3)
        cv2.putText(frame, str(gaze_ratio_du), (50, 50), font, 2, (0, 0, 255), 3)


        
        if gaze_ratio_lr == 1:  #1 e centar 
            new_frame[:] = (0, 255, 0)  #green
            cv2.putText(frame, "CENTER", (50, 150), font, 2, (0, 0, 255), 3)
            
        elif gaze_ratio_lr == 0:  #0 e levo
            new_frame[:] = (0, 0, 255) #new frame za bojata e toj frame, blue e 255
            cv2.putText(frame, "LEFT", (50, 150), font, 2, (0, 0, 255), 3)  #za tekstot specifikacii
            # x = requests.post(url,data ='{"direction": "RIGHT"}')
        elif gaze_ratio_lr == 2:  #2 e desno 
            new_frame[:] = (255, 0, 0)  #red
            cv2.putText(frame, "RIGHT", (50, 150), font, 2, (0, 0, 255), 3)
            
        if gaze_ratio_du == 1:  #1 e centar
            #new_frame[:] = (0, 255, 0)  #red
            cv2.putText(frame, "CENTER", (150, 75), font, 2, (0, 0, 255), 3)
        elif gaze_ratio_du == 0:  #0 e dolu
            #new_frame[:] = (0, 0, 255) #new frame za bojata e toj frame, blue e 255
            cv2.putText(frame, "DOWN", (150, 75), font, 2, (0, 0, 255), 3)  #za tekstot specifikacii
            # x = requests.post(url,data ='{"direction": "RIGHT"}')
        elif gaze_ratio_du == 2:  #2 e gore
            #new_frame[:] = (255, 0, 0)  #red
            cv2.putText(frame, "UP", (150, 75), font, 2, (0, 0, 255), 3)
        
        
        
  
                            
    cv2.imshow("Frame", frame)
    cv2.imshow("New frame", new_frame)

    key = cv2.waitKey(1)
    #list_iter_pos = (list_iter_pos + 1) % 10
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
