#---------------------------------------
#                BIOBOT
#---------------------------------------
# 1. Jazmín Alondra Urías García
# 2. Carlos Eduardo Rosales Pineda
# 3. Brian Antonio Soto Loaiza
# 4. Carlos Jesus Ritchie Aviles
#---------------------------------------

#Librerías
import cv2
import numpy as np

# Inicializa la cámara
cam = cv2.VideoCapture(0)
kernel = np.ones((5, 5), np.uint8)

while True:
    # Captura un frame de la cámara
    ret, frame = cam.read()

    # Frame de BGR a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definición de rangos de colores
    # Azul
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Rojo
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Verde
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    # Blanco
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 20, 255])

    # Negro
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])

    # Combinar máscaras para diferentes colores
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # Combinar todas las máscaras
    combined_mask = mask_blue + mask_red + mask_green + mask_white + mask_black

    # Operaciones para eliminar el ruido
    opening = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Contornos en la imagen
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Área total de la imagen
    total_area = frame.shape[0] * frame.shape[1]

    # Límites para el área del contorno
    min_area_ratio = 0.003
    max_area_ratio = 0.1

    # Contornos y se muestra en la cap la palabra "OBJETOS"
    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / total_area
        if min_area_ratio < area_ratio < max_area_ratio:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    cv2.putText(frame, 'BioBot--OBJETOS', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Frame
    cv2.imshow('Detención de Objetos', frame)

    # Salir con la tecla Enter
    k = cv2.waitKey(1)
    if k == 13:
        break

# Se cierran todas las ventanas
cam.release()
cv2.destroyAllWindows()
