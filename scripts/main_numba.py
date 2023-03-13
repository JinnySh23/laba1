# -*- coding: utf-8 -*-
# *******************************************************************************
# 
#                               CV Lab 1
#
#                   Author: Evgeny B.A. ITMO University
# 
# _______________________________________________________________________________

import cv2
import numpy as np
import time
from numba import njit

# --------------------------------------------------------------------------------------
#                                   CONSTANTES
# --------------------------------------------------------------------------------------
#Входное видео
VIDEO_PATH = "../media/video.mp4"

# Время остановки кадра для взятия (мс)
FRAME_TIME_MS = 6000

# Функция для рассчёта и получения гистограммы
@njit
def getHistogram(image):

    start_time = time.perf_counter()
    # Рассчитываем гистограмму с помощью функции calcHist()
    hist = cv2.calcHist([image],[0],None,[256],[0,256])
    end_time = time.perf_counter()
    print(f"Время выполнения программы: {(end_time - start_time) * 1000:.2f} мс")

    # Создаем чистое изображение для отображения гистограммы
    hist_img = np.zeros((256,256,3), np.uint8)
    hist_img.fill(255)

    # Нормируем гистограмму
    cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)

    # Определяем максимальное значение гистограммы
    max_val = np.max(hist)

    # Рисуем гистограмму
    for i in range(256):
        h = int(hist[i]*256/max_val)
        cv2.line(hist_img,(i,256),(i,256-h),(0,0,0))

    return hist_img

# Функция для эквализации изображения
def equalizeImage(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # Разделяем L, A и B каналы
    l, a, b = cv2.split(lab)

    # Эквализируем L канал
    equalized_l = cv2.equalizeHist(l)

    # Объединяем L, A и B каналы в одно изображение в LAB пространстве
    equalized_lab = cv2.merge((equalized_l, a, b))

    # Конвертируем изображение в RGB
    equalized_img = cv2.cvtColor(equalized_lab, cv2.COLOR_LAB2RGB)

    return equalized_img

# --------------------------------------------------------------------------------------
#                                   
#                                           MAIN
# 
# --------------------------------------------------------------------------------------

if __name__ == "__main__":

    # Открываем видео
    videocap = cv2.VideoCapture(VIDEO_PATH)

    # Останавливаем кадр
    videocap.set(cv2.CAP_PROP_POS_MSEC,FRAME_TIME_MS)
    
    # Забираем результат с фреймом
    is_success, original_image = videocap.read()

    if is_success:

        # Подгатавливаем окно для показа оригинального изображения и сразу показываем его
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.imshow('Original', original_image)
        cv2.moveWindow('Original', 400, 100)

        # Забираем гистограмму эквализированного изображения
        image_histogram = getHistogram(original_image)

        # Подгатавливаем окно для показа гистограммы оригинального изображения и сразу показываем её
        cv2.namedWindow('Histogram', cv2.WINDOW_NORMAL)
        cv2.imshow('Histogram', image_histogram)
        cv2.moveWindow('Histogram', 100, 100)

        
        # Эквализируем изображение
        equ_imege = equalizeImage(original_image)
        
        # --------------------------------------------------------------------------------------
        #                                   KEYSTROKE_HANDLER
        # --------------------------------------------------------------------------------------

        while True:
            key = cv2.waitKey(1) & 0xFF

            # Переключиться между оригинальным и эквализированным изображением по нажатию кнопки "x"
            if key == ord('x'):
                if cv2.getWindowProperty('Original', cv2.WND_PROP_VISIBLE) == 1:
                    cv2.destroyWindow('Original')
                    cv2.namedWindow('Equalized', cv2.WINDOW_NORMAL)
                    cv2.imshow('Equalized', equ_imege)
                    cv2.moveWindow('Equalized', 400, 100)
                    image_histogram = getHistogram(equ_imege)
                    cv2.namedWindow('Histogram', cv2.WINDOW_NORMAL)
                    cv2.imshow('Histogram', image_histogram)
                    cv2.moveWindow('Histogram', 100, 100)
        
                else:
                    cv2.destroyWindow('Equalized')
                    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
                    cv2.imshow('Original', original_image)
                    cv2.moveWindow('Original', 400, 100)
                    image_histogram = getHistogram(original_image)
                    cv2.namedWindow('Histogram', cv2.WINDOW_NORMAL)
                    cv2.imshow('Histogram', image_histogram)
                    cv2.moveWindow('Histogram', 100, 100)

            # Выход из цикла, если нажата клавиша "q"
            elif key == ord('q'):
                print("Done!")
                break

    else:
        print("Ошибка загрузки видео")
        
        # Освободить ресурсы
        videocap.release()
        cv2.destroyAllWindows()
        exit()

    # Освободить ресурсы
    videocap.release()
    cv2.destroyAllWindows()