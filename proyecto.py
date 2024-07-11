import cv2
import torch
import pytesseract
from gtts import gTTS
import os
import speech_recognition as sr
from geopy.geocoders import Nominatim
import googlemaps

# Configurar la ruta de Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.TESSDATA_PREFIX = r'C:\Program Files\Tesseract-OCR\tessdata'

# Configurar el modelo YOLOv5
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Configurar la API de Google Maps
gmaps = googlemaps.Client(key='AIzaSyDi2W3ANQWSQ5vrnSaiAxAO0b2Dyco0iKA')  # Reemplaza con tu propia API key

# Inicializar el reconocedor de voz
r = sr.Recognizer()

# Función para reproducir texto como voz
def reproducir_texto(texto):
    speech = gTTS(text=texto, lang='es', slow=False)
    archivo_audio = "texto.mp3"
    speech.save(archivo_audio)
    os.system(f"start {archivo_audio}")
    os.remove(archivo_audio)  # Eliminar el archivo de audio después de reproducir

# Función para obtener la ubicación actual usando geopy
def obtener_ubicacion():
    try:
        geolocator = Nominatim(user_agent="mi_app_de_voz")
        location = geolocator.geocode("Mi ubicación actual")
        return location.latitude, location.longitude
    except Exception as e:
        print(f'Error al obtener la ubicación: {e}')
        return None, None

# Función para detectar objetos usando YOLOv5
def detector_objetos():
    # Inicializar la cámara
    camara = cv2.VideoCapture(0)

    while camara.isOpened():
        status, frame = camara.read()

        if not status:
            break

        # Realizar la inferencia con YOLOv5
        results = model(frame)

        # Obtener los resultados de detección como un DataFrame
        df = results.pandas().xyxy[0]

        # Filtrar por confianza
        df = df[df["confidence"] > 0.5]

        # Dibujar cuadros de texto y etiquetas en el frame
        for i in range(df.shape[0]):
            bbox = df.iloc[i][["xmin", "ymin", "xmax", "ymax"]].astype(int).values
            class_name = df.iloc[i]['name']
            confidence = round(df.iloc[i]['confidence'], 2)

            # Coordenadas y tamaño del cuadro de texto
            x1, y1, x2, y2 = bbox
            text = f"{class_name}, {confidence}"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Dibujar un cuadro azul con bordes redondeados
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 10, y1), (255, 0, 0), -1)  # Relleno azul
            cv2.putText(frame, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Texto blanco

        # Mostrar el frame con las detecciones
        cv2.imshow("Detección de Objetos", frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar la ventana
    camara.release()
    cv2.destroyAllWindows()

# Función para detectar texto usando Tesseract OCR
def detector_texto():
    # Inicializar la cámara
    camara = cv2.VideoCapture(0)

    while camara.isOpened():
        status, frame = camara.read()

        if not status:
            break

        # Convertir el frame a escala de grises para OCR
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Realizar OCR para detectar texto en la imagen
        ocr_text = pytesseract.image_to_string(gray_frame, lang='spa')

        # Si se detecta texto, mostrarlo y decirlo en voz alta
        if ocr_text.strip():
            print(f"Texto detectado: {ocr_text.strip()}")
            reproducir_texto(ocr_text.strip())

        # Mostrar el frame
        cv2.imshow("Detección de Texto", frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar la ventana
    camara.release()
    cv2.destroyAllWindows()

# Función para manejar la entrada de voz y seleccionar la acción
def manejar_entrada_de_voz():
    while True:
        try:
            with sr.Microphone() as source:
                print("Dime qué quieres hacer...")
                audio = r.listen(source, timeout=None)

            texto = r.recognize_google(audio, language="es-ES")
            print(f"Usuario dijo: {texto}")

            # Procesar la intención del usuario
            if "detectar objetos" in texto.lower():
                detector_objetos()
            elif "detectar texto" in texto.lower():
                detector_texto()
            elif "ubicación actual" in texto.lower():
                lat, lon = obtener_ubicacion()
                if lat is not None and lon is not None:
                    direccion = obtener_direccion(lat, lon)
                    respuesta = f"Estás en {direccion}."
                    reproducir_texto(respuesta)
                    print(respuesta)
                else:
                    reproducir_texto("No pude obtener tu ubicación actual.")
            elif "salir" in texto.lower():
                break
            else:
                reproducir_texto("Lo siento, no entendí lo que dijiste.")

        except sr.UnknownValueError:
            print("Google Speech Recognition no entendió el audio")
        except sr.RequestError as e:
            print(f"Error al solicitar el reconocimiento de voz; {e}")
        except Exception as e:
            print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    manejar_entrada_de_voz()
