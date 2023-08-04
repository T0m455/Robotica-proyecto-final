#!/usr/bin/env python

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import math
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
import cv2
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckietown-udem1')
parser.add_argument('--map-name', default=None)
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

# Constantes
DUCKIE_MIN_AREA = 0
RED_LINE_MIN_AREA = 0
RED_COLOR = (0,0,255)
LINES_COLOR = {"white": (255,255,255), "yellow": (0,215,255)}

# Variables para detención en rojo
init_time = 0
stop_count = 0

# === PID CONTROLLER =================================================================================
Kp=0.20
Kd=0
Ki=0.10

x_objetivo=3
y_objetivo=0.8
x_inicial=1.2
y_inicial=0.8

init_angle = 0

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_vel = 0
        self.previous_error = 0
        self.integral_rot = 0
        self.previous_error_rot = 0
        self.previous_error_vel = 0

    def update_err(self, error_vel, error_rot):
        self.integral_vel += error_vel
        self.integral_rot += error_rot
        derivative_vel = error_vel - self.previous_error_vel
        derivative_rot = error_rot - self.previous_error_rot
        control_rot = self.Kp * error_rot + self.Ki * self.integral_rot + self.Kd * derivative_rot
        control_vel = self.Kp * error_vel + self.Ki * self.integral_vel + self.Kd * derivative_vel
        self.previous_error_rot = error_rot
        self.previous_error_vel = error_vel
        return control_vel, control_rot

    def reset_error(self):
        self.previous_error=0
        self.integral=0

def PID_controller(xt, yt, xf,yf):

    distance_to_target = np.sqrt((xf - xt) ** 2 + (yf - yt) ** 2)
    angle_difference = np.arctan2(yf - yt, xf - xt)
    angle_difference = np.arctan2(np.sin(angle_difference), np.cos(angle_difference))  # Normalizar angulo

    # Calcular el error del controlador PID
    linear_error = distance_to_target * np.cos(angle_difference)
    angular_error = -angle_difference
    
    # Calcular controles usando PID
    linear_velocity, angular_velocity= PIDControl.update_err(linear_error, angular_error)

    return linear_velocity, angular_velocity

# ====================================================================================================

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
    )
    PIDControl= PIDController(Kp,Ki,Kd)
else:
    env = gym.make(args.env_name)

env.reset()
env.cur_pos=[x_inicial,0,y_inicial] # Posición inicial de duckiebot
env.cur_angle=init_angle            # Angulo inicial de duckiebot
env.render(mode="top_down")

# Funciones básicas para el procesamiento posterior
def get_angle_degrees(line):
    x1, y1, x2, y2 = line
    ret_val = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
    if ret_val < 0:
        return 180.0 + ret_val
    return ret_val

def get_angle_radians(line):
    x1, y1, x2, y2 = line
    ret_val = math.atan2(y2 - y1, x2 - x1)
    if ret_val < 0:
        return math.pi + ret_val
    return ret_val

def line_intersect(line1, line2):
    """ returns a (x, y) tuple or None if there is no intersection """
    ax1, ay1, ax2, ay2 = line1
    bx1, by1, bx2, by2 = line2

    d = (by2 - by1) * (ax2 - ax1) - (bx2 - bx1) * (ay2 - ay1)
    if d:
        uA = ((bx2 - bx1) * (ay1 - by1) - (by2 - by1) * (ax1 - bx1)) / d
        uB = ((ax2 - ax1) * (ay1 - by1) - (ay2 - ay1) * (ax1 - bx1)) / d
    else:
        return None, None
    if not (0 <= uA <= 1 and 0 <= uB <= 1):
        return None, None
    x = ax1 + uA * (ax2 - ax1)
    y = ay1 + uA * (ay2 - ay1)

    return x, y

# ====================================================================================================

def duckie_detection(obs, converted, frame):
    '''
    Detectar patos, retornar si hubo detección y el ángulo de giro en tal caso 
    para lograr esquivar el duckie y evitar la colisión.
    '''

    # Se asume que no hay detección
    detection = False
    angle = 0

    # Implementar filtros --------------------------------------------------------------------------
    filtro_1 = np.array([15, 240, 100])  # rango minimo para amarillo/naranja en HSV
    filtro_2 = np.array([30, 255, 255])  # rango máximo para amarillo/naranja en HSV

    mask_duckie = cv2.inRange(converted, filtro_1, filtro_2)  # Aplicar mascara para filtrar duckie

    # Segmentación con operaciones morfológicas (erode y dilate) -----------------------------------
    kernel = np.ones((5,5),np.uint8)             # Crear kernel para operaciones morfologicas

    image_out = cv2.erode(mask_duckie, kernel, iterations = 2)  # Operacion morfologica erode   
    image_out = cv2.dilate(image_out, kernel, iterations = 10)  # Operacion morfologica dilate

    segment_image = cv2.bitwise_and(converted, converted, mask=mask_duckie)  # Aplicar segmentación

    contours, hierarchy = cv2.findContours(image_out, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

     # Observar la imagen post-opening
    segment_image_post_opening = cv2.bitwise_and(converted, converted, mask= image_out)
    segment_image_post_opening =  cv2.cvtColor(segment_image_post_opening, cv2.COLOR_HSV2BGR)

    # Revisar los contornos identificados y dibujar el rectángulo correspondiente ------------------

    for cnt in contours:
        # Obtener rectangulo
        x, y, w, h = cv2.boundingRect(cnt)
        duckie_box_area = w*h

        # Filtrar por area minima
        if duckie_box_area > DUCKIE_MIN_AREA:

            x2 = x + w  # obtener el otro extremo
            y2 = y + h

            # Dibujar un rectangulo en la imagen
            cv2.rectangle(frame, (int(x), int(y)), (int(x2),int(y2)), (0,255,0), 3)
            # Si la distancia es muy pequeña el duckiebot debe evitar la colision
            # Para esto medimos la distancia al duckie por la altura y area de la bounding box
            # Entonces así definimos un rango donde veamos que llega a "chocar"
           
            if duckie_box_area > 7500 and h > 300:
                cv2.rectangle(frame, (int(x), int(y)), (int(x2),int(y2)), (255,255,255), 3)
                detection = True
                angle = 1
    
    # Mostrar ventanas con los resultados ----------------------------------------------------------
    #cv2.imshow("Patos filtro", segment_image_post_opening)
    #cv2.imshow("Patos detecciones", frame)

    return detection, angle, frame

def duckie_detection_alien(obs, converted, frame):
    '''
    Detectar patos, retornar si hubo detección y el ángulo de giro en tal caso 
    para lograr esquivar el duckie y evitar la colisión.
    '''

    # Se asume que no hay detección
    detection = False
    angle = 0

    # Implementar filtros --------------------------------------------------------------------------
    filtro_1 = np.array([56, 255, 0])  # rango minimo para amarillo/naranja en HSV
    filtro_2 = np.array([72, 255, 255])  # rango máximo para amarillo/naranja en HSV

    mask_duckie = cv2.inRange(converted, filtro_1, filtro_2)  # Aplicar mascara para filtrar duckie

    # Segmentación con operaciones morfológicas (erode y dilate) -----------------------------------
    kernel = np.ones((5,5),np.uint8)             # Crear kernel para operaciones morfologicas

    image_out = cv2.erode(mask_duckie, kernel, iterations = 2)  # Operacion morfologica erode   
    image_out = cv2.dilate(image_out, kernel, iterations = 10)  # Operacion morfologica dilate

    segment_image = cv2.bitwise_and(converted, converted, mask=mask_duckie)  # Aplicar segmentación

    contours, hierarchy = cv2.findContours(image_out, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

     # Observar la imagen post-opening
    segment_image_post_opening = cv2.bitwise_and(converted, converted, mask= image_out)
    segment_image_post_opening =  cv2.cvtColor(segment_image_post_opening, cv2.COLOR_HSV2BGR)

    # Revisar los contornos identificados y dibujar el rectángulo correspondiente ------------------

    for cnt in contours:
        # Obtener rectangulo
        x, y, w, h = cv2.boundingRect(cnt)
        duckie_box_area = w*h

        # Filtrar por area minima
        if duckie_box_area > DUCKIE_MIN_AREA:

            x2 = x + w  # obtener el otro extremo
            y2 = y + h

            # Dibujar un rectangulo en la imagen
            cv2.rectangle(frame, (int(x), int(y)), (int(x2),int(y2)), (0,0,255), 3)
            # Si la distancia es muy pequeña el duckiebot debe evitar la colision
            # Para esto medimos la distancia al duckie por la altura y area de la bounding box
            # Entonces así definimos un rango donde veamos que llega a "chocar"
           
            if duckie_box_area > 7500 and h > 300:
                cv2.rectangle(frame, (int(x), int(y)), (int(x2),int(y2)), (255,255,255), 3)
                detection = True
                angle = 1
    
    # Mostrar ventanas con los resultados ----------------------------------------------------------
    #cv2.imshow("Patos filtro", segment_image_post_opening)
    #cv2.imshow("Patos detecciones", frame)

    return detection, angle, frame


def red_line_detection(converted, frame):
    '''
    Detección de líneas rojas en el camino, esto es análogo a la detección de duckies.
    '''
    # Se asume que no hay detección
    detection = False

    # Cortar imagen para obtener zona relevante
    height, width = converted.shape[:2]
    img = np.zeros_like(converted)
    img[260:640,0:640] = converted[260:640,0:640]

    # Implementar filtro
    red_filter1 = np.array([175, 100, 20])  # rango minimo para rojo en HSV
    red_filter2 = np.array([179, 255, 255]) # rango máximo para rojo en HSV

    mask_line = cv2.inRange(img, red_filter1, red_filter2)  # Aplicar mascara para filtrar

    # Segmentación con operaciones morfológicas (erode y dilate) -----------------------------------
    kernel = np.ones((5,5),np.uint8) # Crear kernel para operaciones morfologicas

    image_out = cv2.erode(mask_line, kernel, iterations = 2)  # Operacion morfologica erode   
    image_out = cv2.dilate(image_out, kernel, iterations = 10)  # Operacion morfologica dilate

    segment_image = cv2.bitwise_and(img, img, mask=mask_line)  # Aplicar segmentación

    contours, hierarchy = cv2.findContours(image_out, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Revisar los contornos identificados y dibujar el rectángulo correspondiente
    for cnt in contours:
        # Obtener rectangulo
        x, y, w, h = cv2.boundingRect(cnt)
        red_line_area = w*h

        # Filtrar por area minima
        if red_line_area > RED_LINE_MIN_AREA:
            x2 = x + w 
            y2 = y + h

            # Dibujar un rectangulo en la imagen
            cv2.rectangle(segment_image, (int(x), int(y)), (int(x2),int(y2)), RED_COLOR, 3)
            print("RED AREA, H, W: ", red_line_area, h, w)

            if red_line_area > 50000 and (h > 80 and w > 520): # Detención
                detection = True
        
    # Mostrar ventanas con los resultados
    #cv2.imshow("Red lines", segment_image)
    return detection, segment_image

def get_line(converted, filter_1, filter_2, line_color):
    '''
    Detecta las lineas del carril y determina su ángulo.
    '''

    # Cortar imagen para obtener zona relevante
    height, width = converted.shape[:2]
    img = np.zeros_like(converted)
    img[260:640, 0:640] = converted[260:640, 0:640]

    # Crear máscara de filtro 
    mask = cv2.inRange(img, filter_1, filter_2)
    segment_image = cv2.bitwise_and(img, img, mask=mask)
    
    # Erosionar la imagen
    image = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)
    kernel = np.ones((5, 5), np.uint8)
    image_lines = cv2.erode(image, kernel, iterations=2)    

    # Detectar bordes
    gray_lines = cv2.cvtColor(image_lines, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_lines, 80, 80, None, 3)
   
    # Detectar líneas usando HoughLines
    rho = 1
    theta = np.pi/180
    threshold = 30

    lines = cv2.HoughLines(edges, rho, theta, threshold)

    angle = [None, None]
    line_points = (0, 0, 0, 0)

    if lines is not None:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a)) 

            line_points = (x1, y1, x2, y2)
            angle = [get_angle_radians(line_points), 
                     get_angle_degrees(line_points)]
            cv2.line(image_lines, (x1, y1), (x2, y2), LINES_COLOR[line_color], 2)

    #cv2.imshow(line_color, image_lines)
    return angle, line_points, image_lines


def line_follower(vel, angle, obs):
    '''
    Controlador para navegar de forma autónoma por el carril.
    '''
    converted = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    frame = obs[:, :, [2, 1, 0]]
    frame = cv2.UMat(frame).get()

    x_vel, rot_vel = vel, angle
    x_objetivo, y_objetivo = env.cur_pos[0], env.cur_pos[2]
    

    # Detección de duckies ----------------------------------------------------------------------------
    pedestrian_detection, duckie_angle, duckie_frame = duckie_detection(obs=obs, frame=frame, converted=converted)
    pedestrian_detection, duckie_angle, duckie_frame = duckie_detection_alien(obs=obs, frame=frame, converted=converted)
    if pedestrian_detection:
        print("DUCKIE!")
        x_vel = 0
        rot_vel += duckie_angle
        return np.array([abs(x_vel), rot_vel])

    # Detección de líneas rojas ------------------------------------------------------------------------
    stop_detection, red_line_frame = red_line_detection(converted=converted, frame=frame)

    global init_time
    global stop_count # Por 5 frames luego de haberse detenido en rojo, no se detectará esta línea

    if stop_count != 0:
        stop_count -= 1

    if stop_detection and stop_count == 0:
        print("STOP")
        current_time = time.time()

        if init_time == 0:
            init_time = current_time

        if (current_time - init_time) <= 3: 
            # Si aun no pasan 3 segundos desde que detecto el rojo, espera
            print(f"Waiting red line... ({round(current_time - init_time, 2)}[s])")
            x_vel = 0
            rot_vel = 0
            return np.array([abs(x_vel), rot_vel])
        else: # Si pasaron 3 segundos desde que detecto el rojo, continua
            init_time = 0
            x_vel += 0.44
            stop_count = 5
            return np.array([abs(x_vel), rot_vel])

    # Detección de líneas del carril -------------------------------------------------------------------

    # Filtros para el detector de lineas blancas
    white_filter_1 = np.array([0, 0, 150])
    white_filter_2 = np.array([180, 60, 255])

    # Filtros para el detector de lineas amarillas
    yellow_filter_1 = np.array([20, 50, 100])
    yellow_filter_2 = np.array([30, 225, 255])

    angle_white, line_white, road1 = get_line(converted, white_filter_1, white_filter_2, "white")
    angle_yellow, line_yellow, road2 = get_line(converted, yellow_filter_1, yellow_filter_2, "yellow")

    road_frame = cv2.addWeighted(road1, 1, road2, 1, 0)
    road_detections = cv2.addWeighted(road_frame, 1, red_line_frame, 1, 0)
    all_detections = cv2.addWeighted(road_detections, 0.6, duckie_frame, 0.4, 0)
    #cv2.imshow("Road detections", road_detections) # mostrar detecciones del camino
    cv2.imshow("Detections", all_detections) # mostrar todas las detecciones

    ''' =============================================================================================
        > Controlador para navegar dentro del mapa
        ============================================================================================= '''

    # Si se detectan las lineas del carril
    if line_white != (0, 0, 0, 0) or line_yellow != (0, 0, 0, 0):

        # Se obtiene la intersección de las lineas del carril
        intersect = line_intersect(line_white, line_yellow)

        if intersect[0] != None: # El objetivo será ir hacia esa intersección --------------------------
            print("Insersect")
            x_objetivo, y_objetivo = intersect

            # Se llama al controlador PID para que calcule su proximo movimiento
            pos_x, pos_y = env.cur_pos[0],env.cur_pos[2]
            x_vel, rot_vel = PID_controller(pos_x, pos_y,x_objetivo,y_objetivo)

        else:  # Si no hay intersección, entonces esta en una curva y debe girar
            x_vel = 0
            if angle_yellow[0] != None: # Si hay linea amarilla ----------------------------------------

                if angle_yellow[1] > 50 and angle_yellow[1] < 100: # si esta vertical, continua
                    print(f"> LINEA VERTICAL - AMARILLA ({round(angle_yellow[1], 2)})")
                    rot_vel = 0
                    x_vel += 0.44
                elif angle_yellow[1] <= 20 or angle_yellow[1] > 180: # Si esta en horizontal, giro completo
                    print(f"> GIRO COMPLETO - AMARILLA, ANGULO ({round(angle_yellow[1], 2)})")
                    rot_vel += 0.35
                    x_vel += 0.12

                else: # Si esta inclinado en otro sentido, giro suave
                    print(f"> GIRO SUAVE - AMARILLA ({round(angle_yellow[1], 2)})")
                    rot_vel += 0.15
                    x_vel += 0.2

            else: # Si no hay linea amarilla -----------------------------------------------------------

                if angle_white[1] > 50 and angle_white[1] < 100: # Si esta vertical, continua
                    print(f"> LINEA VERTICAL - BLANCA ({round(angle_white[1], 2)})")
                    rot_vel = 0
                    x_vel += 0.44
                elif angle_white[1] <= 50 or angle_white[1] > 180: # Si esta en horizontal, giro completo
                    print(f"> GIRO COMPLETO - BLANCA ({round(angle_white[1], 2)})")
                    rot_vel -= 0.35
                    x_vel += 0.12

                else: # Si esta inclinado en otro sentido, giro suave
                    print(f"> GIRO SUAVE - BLANCA ({round(angle_white[1], 2)})")
                    rot_vel -= 0.15
                    x_vel += 0.2

    else: # Si no se detectan lineas sigue de largo ----------------------------------------------------
        print("No Lines")
        x_vel += 0.44

    vel, angle = np.array([abs(x_vel), rot_vel])

    return np.array([vel, angle]) # Implementar nuevo ángulo de giro controlado

# ====================================================================================================

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    Handler para reiniciar el ambiente
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.cur_pos=[x_inicial, 0, y_inicial]
        env.cur_angle=init_angle
        env.render(mode="top_down")

    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0

    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)


# Registrar el handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

action = np.array([0.44, 0.0])

def update(dt):
    """
    Funcion que se llama en step.
    """
    global action
    # Aquí se controla el duckiebot
    #action = np.array([0.0, 0.0]) # Activar para control manual

    if key_handler[key.UP]:
        action[0] += 0.44
    if key_handler[key.DOWN]:
        action[0] -= 0.44
    if key_handler[key.LEFT]:
        action[1] += 1
    if key_handler[key.RIGHT]:
        action[1] -= 1
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    # obs consiste en un imagen de 640 x 480 x 3
    obs, reward, done, info = env.step(action)
    vel, angle = line_follower(action[0], action[1], obs)
    action[0], action[1] = vel, angle # Comentar para control manual
    
    if done:
        print('done!')
        env.reset()
        env.cur_pos=[x_inicial, 0, y_inicial]
        env.cur_angle=init_angle
        env.render(mode="top_down")

    cv2.waitKey(1)
    env.render(mode="top_down")

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()