import argparse
import logging
import time
from pprint import pprint
import cv2
import numpy as np
import sys
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import math

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
def find_point(pose, p):
    for point in pose:
        try:
            body_part = point.body_parts[p]
            return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
        except:
            return (0,0)
    return (0,0)
def euclidian( point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 )
def angle_calc(p0, p1, p2 ):
    '''
        p1 is center point from where we measured angle between p0 and p2
    '''
    try:
        a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
        b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
        c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
        angle = math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180/math.pi
    except:
        return 0
    return int(angle)
def plank( a, b, c, d, e, f):
    #There are ranges of angle and distance to for plank.
    '''
        a and b are angles of hands
        c and d are angle of legs
        e and f are distance between head to ankle because in plank distace will be maximum.
    '''
    if (a in range(50,100) or b in range(50,100)) and (c in range(135,175) or d in range(135,175)) and (e in range(50,250) or f in range(50,250)):
        return True
    return False
def lateral_elevation_phase_1(hand_to_hand_distance, shoulder_angle_l, shoulder_angle_r, head_hand_dst_r, head_hand_dst_l):
    '''
        m_pose is distance between two wrists
        angle1 and angle5 are angle between neck,shoulder and wrist
        head_hand_dst_r and head_hand_dst_l are distance between head to hands.
    '''
    if hand_to_hand_distance in range(80, 250) and shoulder_angle_l in range(100, 150) and shoulder_angle_r in range(100, 150) and head_hand_dst_r > 160 and head_hand_dst_l > 160 :
        return True
    return False
def lateral_elevation_phase_2(hand_to_hand_distance, shoulder_angle_l, shoulder_angle_r, head_hand_dst_r, head_hand_dst_l):
    '''
        m_pose is distance between two wrists
        angle1 and angle5 are angle between neck,shoulder and wrist
        head_hand_dst_r and head_hand_dst_l are distance between head to hands.
    '''
    #and head_hand_dst_r in range(100, 150) and head_hand_dst_l in range(100, 150)
    #hand_to_hand_distance in range(301, 500) and
    if shoulder_angle_l in range(120, 160) and shoulder_angle_r in range(120, 160):
        return True
    return False
def lateral_elevation_phase_3(hand_to_hand_distance, shoulder_angle_l, shoulder_angle_r, head_hand_dst_r, head_hand_dst_l):
    '''
        m_pose is distance between two wrists
        angle1 and angle5 are angle between neck,shoulder and wrist
        head_hand_dst_r and head_hand_dst_l are distance between head to hands.
    '''

    if shoulder_angle_l in range(161, 190) and shoulder_angle_r in range(161, 190):
        return True
    return False
def draw_str(dst, coordinates, s, color, scale):

    (x, y) = coordinates
    if (color[0]+color[1]+color[2]==255*3):
        cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, scale, (0, 0, 0), thickness = 4, lineType=10)
    else:
        cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness = 4, lineType=10)
    #cv2.line
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, (255, 255, 255), lineType=11)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=str, default=0)
    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=432x368, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    print("modo 0: Mapeamento apenas \nmodo 1: Elevação Lateral \nmode 2: prancha")
    mode = int(input("Selecione o modo : "))

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        poseEstimator = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        poseEstimator = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800);
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600);
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    count = 0
    i = 0
    frm = 0
    right_hand_prev = [0,0]
    left_hand_prev = [0,0]
    global height,width
    orange_color = (0,140,255)
    green_color = (0,255,0)
    red_color = (0,0,255)
    white_color = (255 ,255,255)
    action = 'Desconhecido'
    reps = 0
    right_error = False
    left_error = False
    while True:
        ret_val, image = cam.read()
        i =1
        humans = poseEstimator.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        pose = humans
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        height,width = image.shape[0],image.shape[1]
        if mode == 1:
            # cv2.putText(image,"Elevação escolhida", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,255), 2, 11)
            if len(pose) > 0:
                # distance calculations
                head_hand_dst_l = int(euclidian(find_point(pose, 0), find_point(pose, 7)))# Cabeça braço direito
                head_hand_dst_r = int(euclidian(find_point(pose, 0), find_point(pose, 4)))# Cabeça braço esquerdo
                hand_to_hand_distance = int(euclidian(find_point(pose, 7), find_point(pose, 4))) # distância entre mãos
                # angle calcucations
                shoulder_angle_l =  angle_calc(find_point(pose, 6), find_point(pose, 5), find_point(pose, 1)) # angulo ombro esquerdo
                shoulder_angle_r =  angle_calc(find_point(pose, 3), find_point(pose, 2), find_point(pose, 1)) # angulo ombro direito
                elbow_angle_l =  angle_calc(find_point(pose, 5), find_point(pose, 6), find_point(pose, 7)) # angulo cptovelo esquerdo
                elbow_angle_r =  angle_calc(find_point(pose, 2), find_point(pose, 3), find_point(pose, 4)) # angulo cptovelo direito

                right_hand_point = find_point(pose, 4)   # ponto da mão direita atualmente
                left_hand_point = find_point(pose, 7)    # ponto da mão esquerda atualmente
                right_hand_prev.append(right_hand_point[1])
                left_hand_prev.append(left_hand_point[1])
                # logger.debug("*** Entrou aqui ***")
                if (mode == 1) and reps == 2:
                    action = 'fim'
                if (mode == 1) and action == 'Desconhecido':
                    #if prev_action == 'Unknown' or prev_action == "Unknown_First":
                    #    exercise_duration = time.time()
                    draw_str(image, (20, 50), "Assuma a posicao inicial de elevacao", orange_color, 2)
                if (mode == 1) and lateral_elevation_phase_1(hand_to_hand_distance, shoulder_angle_l, shoulder_angle_r, head_hand_dst_r, head_hand_dst_l) and (action == 'Desconhecido' or action == 'fase4'):
                    # incremente o contador aqui
                    if action == "fase4":
                        reps+=1
                    action = "fase1"
                    draw_str(image, (20, 50), "Eleve os bracos mantendo o angulo", green_color, 2)
                if (mode == 1) and action == "fase1":
                    test = left_hand_point[1] - right_hand_point[1]
                    draw_str(image, (20, 100),  str(test), red_color, 4)
                    # if m_pose in range(20,300) and angle1 in range(60,140) and angle5 in range(60,140) and head_hand_dst_r in range(100,145) and head_hand_dst_l in range(100,145):
                    if ((right_hand_point[1] - right_hand_prev[-2]) > 1) or ((right_hand_point[1] - left_hand_point[1]) > 6):  # it's distance between frame and comparing it with thresold value
                        draw_str(image, (20, 50), "Elevar mao direita", red_color, 2)
                    elif ((left_hand_point[1] - left_hand_prev[-2]) > 1) or ((left_hand_point[1] - right_hand_point[1]) > 6):  # it's distance between frame and comparing it with thresold value
                        draw_str(image, (20, 50), "Elevar mao esquerda", red_color, 2)
                    elif not (hand_to_hand_distance in range(70, 300)):
                        draw_str(image, (20, 50), "Continue elevando os bracos", red_color, 2)
                    elif not (elbow_angle_l in range(130, 170)):
                        draw_str(image, (20, 50), "Cotovelo esquerdo fora do angulo adequado", red_color, 2)
                    elif not (elbow_angle_r in range(130, 170)):
                        draw_str(image, (20, 50), "Cotovelo direito fora do angulo adequado", red_color, 2)
                    else:
                        draw_str(image, (20, 50), "Eleve os bracos mantendo o angulo", green_color, 2)
                if (mode == 1) and lateral_elevation_phase_2(hand_to_hand_distance, shoulder_angle_l, shoulder_angle_r, head_hand_dst_r, head_hand_dst_l) and action == 'fase1':
                    action = "fase2"
                    draw_str(image, (20, 50), "Eleve os braços até ficarem \nparalelos ao solo", green_color, 2)
                if (mode == 1) and action == "fase2":
                    test = left_hand_point[1] - right_hand_point[1]
                    draw_str(image, (20, 100),  str(test), red_color, 4)
                    if ((right_hand_point[1] - right_hand_prev[-2]) > 1) or ((right_hand_point[1] - left_hand_point[1]) > 6):  # it's distance between frame and comparing it with thresold value
                        draw_str(image, (20, 50), "Elevar mao direita", red_color, 2)
                    elif ((left_hand_point[1] - left_hand_prev[-2]) > 1) or ((left_hand_point[1] - right_hand_point[1]) > 6):  # it's distance between frame and comparing it with thresold value
                        draw_str(image, (20, 50), "Elevar mao esquerda", red_color, 2)
                    # if m_pose in range(20,300) and angle1 in range(60,140) and angle5 in range(60,140) and head_hand_dst_r in range(100,145) and head_hand_dst_l in range(100,145):
                    elif not (elbow_angle_l in range(150, 190)):
                        draw_str(image, (20, 50), "Cotovelo esquerdo fora do angulo adequado", red_color, 2)
                    elif not (elbow_angle_r in range(150, 190)):
                        draw_str(image, (20, 50), "Cotovelo direito fora do angulo adequado", red_color, 2)
                    else:
                        draw_str(image, (20, 50), "Eleve os bracos ate ficarem paralelos ao solo", green_color, 2)
                if (mode == 1) and lateral_elevation_phase_3(hand_to_hand_distance, shoulder_angle_l, shoulder_angle_r, head_hand_dst_r, head_hand_dst_l) and action == 'fase2':
                    action = "fase3"
                    draw_str(image, (20, 50), "Desca os bracos mantendo o angulo correto", green_color, 2)
                if (mode == 1) and action == "fase3":
                    # if m_pose in range(20,300) and angle1 in range(60,140) and angle5 in range(60,140) and head_hand_dst_r in range(100,145) and head_hand_dst_l in range(100,145):
                    if ((right_hand_prev[-2] - right_hand_point[1]) > 1) or ((left_hand_point[1] - right_hand_point[1]) > 6):  # it's distance between frame and comparing it with thresold value
                        draw_str(image, (20, 50), "Descer mao direita", red_color, 2)
                    elif ((left_hand_prev[-2] - left_hand_point[1]) > 1) or ((right_hand_point[1] - left_hand_point[1]) > 6):  # it's distance between frame and comparing it with thresold value
                        draw_str(image, (20, 50), "Descer mao esquerda", red_color, 2)
                    elif not (elbow_angle_l in range(150, 190)):
                        draw_str(image, (20, 50), "Cotovelo esquerdo fora do angulo adequado", red_color, 2)
                    elif not (elbow_angle_r in range(150, 190)):
                        draw_str(image, (20, 50), "Cotovelo direito fora do angulo adequado", red_color, 2)
                    else:
                        draw_str(image, (20, 50), "Desca os bracos mantendo o angulo correto", green_color, 2)
                if (mode == 1) and lateral_elevation_phase_2(hand_to_hand_distance, shoulder_angle_l, shoulder_angle_r, head_hand_dst_r, head_hand_dst_l) and action == 'fase3':
                    action = "fase4"
                    draw_str(image, (20, 50), "Retorne para a posicao inicial", green_color, 2)
                if (mode == 1) and action == "fase4":
                    # if m_pose in range(20,300) and angle1 in range(60,140) and angle5 in range(60,140) and head_hand_dst_r in range(100,145) and head_hand_dst_l in range(100,145):
                    if ((right_hand_prev[-2] - right_hand_point[1]) > 1) or ((left_hand_point[1] - right_hand_point[1]) > 6):  # it's distance between frame and comparing it with thresold value
                        draw_str(image, (20, 50), "Descer mao direita", red_color, 2)
                    elif ((left_hand_prev[-2] - left_hand_point[1]) > 1) or ((right_hand_point[1] - left_hand_point[1]) > 6):  # it's distance between frame and comparing it with thresold value
                        draw_str(image, (20, 50), "Descer mao esquerda", red_color, 2)
                    if not (elbow_angle_l in range(160, 190)):
                        draw_str(image, (20, 50), "Cotovelo esquerdo fora do angulo adequado", red_color, 2)
                    elif not (elbow_angle_r in range(160, 190)):
                        draw_str(image, (20, 50), "Cotovelo direito fora do angulo adequado", red_color, 2)
                    else:
                        draw_str(image, (20, 50), "Retorne para a posicao inicial", green_color, 2)
                if (mode == 1) and action == "fim":
                    draw_str(image, (50, 100), "Exercicio concluido", green_color, 3)
            if len(pose) == 0 :
                action = 'Desconhecido'
                reps = 0
            draw_str(image, (10, 450), "repeticoes = "+ str(reps), white_color, 2)
            logger.debug("*** Elevacao lateral *** - "+action)
        elif mode == 2:
            if len(pose) > 0:
                # distance calculations
                head_hand_dst_l = int(euclidian(find_point(pose, 0), find_point(pose, 7)))
                head_hand_dst_r = int(euclidian(find_point(pose, 0), find_point(pose, 4)))
                # angle calcucations
                angle2 =  angle_calc(find_point(pose,7), find_point(pose,6), find_point(pose,5))
                angle4 =  angle_calc(find_point(pose,11), find_point(pose,12), find_point(pose,13))
                angle6 =  angle_calc(find_point(pose,4), find_point(pose,3), find_point(pose,2))
                angle8 =  angle_calc(find_point(pose,8), find_point(pose,9), find_point(pose,10))

                if (mode == 2) and plank(angle2, angle6, angle4, angle8,head_hand_dst_r, head_hand_dst_l):
                    action = "Plank"
                    #if prev_action == 'Unknown' or prev_action == "Unknown_First":
                    #    exercise_duration = time.time()
                    #logger.debug("*** Plank ***")
                    draw_str(image, (20, 50), " Prancha", orange_color, 2)
                    logger.debug("*** Plank ***")
                    #cv2.putText(image,"PLANK", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,255), 2, 11)

        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        #image =   cv2.resize(image, (720,720))
        if(frm==0):
            out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (image.shape[1],image.shape[0]))
            print("Initializing")
            frm+=1
        cv2.imshow('tf-pose-estimation result', image)
        if i != 0:
            out.write(image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()