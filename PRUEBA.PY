import cv2
from ultralytics import YOLO
from gtts import gTTS
from googletrans import Translator
import os
from time import sleep, time

# Función para obtener los colores de las clases
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

# Función para reproducir texto como voz
def reproducir_texto(texto, lenguaje='es-us'):
    speech = gTTS(text=texto, lang=lenguaje, slow=False)
    archivo_audio = "texto.mp3"
    speech.save(archivo_audio)
    os.system(f"start {archivo_audio}")
    sleep(2)  # Esperar a que se reproduzca completamente antes de eliminarlo
    os.remove(archivo_audio)

if __name__ == "__main__":
    # Inicializar el modelo YOLO
    yolo = YOLO('yolov8s.pt')

    # Iniciar el traductor
    translator = Translator()

    # Iniciar la captura de video desde la cámara
    videoCap = cv2.VideoCapture(0)

    # Tiempo mínimo entre reproducciones de audio (en segundos)
    tiempo_espera_audio = 5
    ultima_reproduccion = 0

    while True:
        ret, frame = videoCap.read()
        if not ret:
            continue
        
        # Realizar la detección de objetos con YOLO
        results = yolo.track(frame, stream=True)

        for result in results:
            # Obtener los nombres de las clases detectadas
            classes_names = result.names

            for box in result.boxes:
                # Verificar la confianza de la detección
                if box.conf[0] > 0.4:
                    # Obtener las coordenadas del cuadro del objeto
                    [x1, y1, x2, y2] = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Obtener la clase del objeto
                    cls = int(box.cls[0])
                    class_name = classes_names[cls]

                    # Traducir el nombre de la clase
                    translated_class_name = translator.translate(class_name, src='en', dest='es').text

                    # Obtener el color correspondiente para la clase
                    colour = getColours(cls)

                    # Dibujar el rectángulo y poner el nombre de la clase y confianza
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(frame, f'{translated_class_name} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

                    # Mostrar el frame con las detecciones
                    cv2.imshow('frame', frame)

                    # Reproducir mensaje de voz si ha pasado el tiempo de espera
                    tiempo_actual = time()
                    if tiempo_actual - ultima_reproduccion > tiempo_espera_audio:
                        objeto = translated_class_name
                        text = f"El objeto detectado es {objeto}."
                        reproducir_texto(text)
                        ultima_reproduccion = tiempo_actual

        # Salir del bucle si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la captura de video y cerrar todas las ventanas
    videoCap.release()
    cv2.destroyAllWindows()

