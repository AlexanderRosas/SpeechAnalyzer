import cv2
import mediapipe as mp
import numpy as np
import threading
import json
import pyaudio
import csv
import os
import sys
from datetime import datetime
from vosk import Model, KaldiRecognizer

# --- CONFIGURACIÓN ---
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 650
TEXT_W_WIDTH = 600
TEXT_W_HEIGHT = 400

MAR_THRESHOLD = 0.5
FROWN_THRESHOLD = 0.06
SMILE_THRESHOLD = 0.45

# --- CLASE DE ANÁLISIS DE SENTIMIENTO ---
class SentimentEngine:
    def __init__(self):
        # Diccionarios básicos para oratoria
        self.positive_words = {
            "feliz", "bueno", "excelente", "alegría", "honor", "ganar", "solución",
            "amor", "esperanza", "futuro", "bien", "gracias", "amigos", "éxito",
            "positivo", "oportunidad", "mejorar", "increíble", "fantástico"
        }
        self.negative_words = {
            "triste", "malo", "terrible", "problema", "error", "miedo", "dolor",
            "crisis", "fracaso", "pérdida", "grave", "lamentable", "odio",
            "injusticia", "peligro", "amenaza", "muerte", "jamás", "nunca"
        }
        self.current_sentiment = "NEUTRAL"
        self.score = 0 # -1 (Negativo) a 1 (Positivo)

    def analyze(self, text):
        if not text: return "NEUTRAL"

        text = text.lower()
        words = text.split()

        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)

        # Lógica simple de balance
        if pos_count > neg_count:
            self.current_sentiment = "POSITIVO"
            return "POSITIVO"
        elif neg_count > pos_count:
            self.current_sentiment = "NEGATIVO"
            return "NEGATIVO"
        else:
            self.current_sentiment = "NEUTRAL"
            return "NEUTRAL"

# --- CLASE DE AUDIO (VOSK) ---
class AudioTranscriber:
    def __init__(self):
        self.last_text = ""
        self.running = True
        self.status = "Iniciando..."
        self.stream = None
        self.model = None
        self.CHUNK = 4096
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "model")

        if not os.path.exists(model_path):
            print(f"--- ERROR: No existe la carpeta: {model_path}")
            self.status = "Error: No hay carpeta 'model'"
            return

        try:
            from vosk import SetLogLevel
            SetLogLevel(-1)
            self.model = Model(model_path)
            self.recognizer = KaldiRecognizer(self.model, self.RATE)
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS,
                                      rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
            self.status = "Escuchando..."
            print("--- VOSK CARGADO ---")
        except Exception as e:
            print(f"--- ERROR VOSK: {e}")
            self.status = "Error Audio"

    def listen_loop(self):
        if self.stream is None: return
        while self.running:
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '')
                    if text: self.last_text = text
                else:
                    partial = json.loads(self.recognizer.PartialResult())
                    partial_text = partial.get('partial', '')
                    if partial_text: self.last_text = partial_text
            except: pass

    def start(self):
        self.thread = threading.Thread(target=self.listen_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.stream: self.stream.stop_stream(); self.stream.close()
        if hasattr(self, 'p'): self.p.terminate()

# --- FUNCIONES AUXILIARES ---
def calculate_distance(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

def get_mar(landmarks):
    h = calculate_distance(landmarks[13], landmarks[14])
    w = calculate_distance(landmarks[61], landmarks[291])
    return h / w if w > 0 else 0

def get_frown_ratio(landmarks): # Detección de ceño
    face_width = calculate_distance(landmarks[234], landmarks[454])
    dist = calculate_distance(landmarks[107], landmarks[336])
    return dist / face_width if face_width > 0 else 1.0

def get_smile_ratio(landmarks): # Detección simple de sonrisa (ancho boca)
    face_width = calculate_distance(landmarks[234], landmarks[454])
    mouth_width = calculate_distance(landmarks[61], landmarks[291])
    return mouth_width / face_width if face_width > 0 else 0

def draw_text_with_wrapping(img, text, font, font_scale, color, thickness, x, y, max_width):
    words = text.split(' ')
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + word + " "
        (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if w < max_width: current_line = test_line
        else: lines.append(current_line); current_line = word + " "
    lines.append(current_line)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + (i * 30)), font, font_scale, color, thickness, cv2.LINE_AA)

# --- MAIN ---
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    if not os.path.exists(data_dir): os.makedirs(data_dir)

    filename = os.path.join(data_dir, f"sesion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    csv_file = None
    writer = None
    try:
        csv_file = open(filename, 'w', newline='', encoding='utf-8')
        writer = csv.writer(csv_file)
        writer.writerow(["Timestamp", "Estado_Facial", "Sentimiento_Texto", "Congruencia", "Texto"])
    except Exception as e: print(e); return

    # Librerías
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    cap = cv2.VideoCapture(0)
    audio_sys = AudioTranscriber()
    sentiment_sys = SentimentEngine() # Instanciamos el motor de sentimientos
    audio_sys.start()

    print("--- SISTEMA CONGRUENCIA ACTIVO ---")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        h_frame, w_frame, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Variables de estado
        face_state = "NEUTRAL"
        text_sentiment = "NEUTRAL"
        congruence_status = "OK" # OK o ALERTA

        mar_value = 0.0
        frown_value = 1.0
        hands_up = False

        # 1. ANÁLISIS DE TEXTO
        text_sentiment = sentiment_sys.analyze(audio_sys.last_text)

        # 2. ANÁLISIS DE IMAGEN
        res_pose = pose.process(image_rgb)
        if res_pose.pose_landmarks:
            lm = res_pose.pose_landmarks.landmark
            if (lm[15].y < lm[11].y) or (lm[16].y < lm[12].y): # Muñecas vs Hombros
                hands_up = True
                face_state = "MANOS ARRIBA"

        res_face = face_mesh.process(image_rgb)
        if res_face.multi_face_landmarks:
            for f_lm in res_face.multi_face_landmarks:
                mar_value = get_mar(f_lm.landmark)
                frown_value = get_frown_ratio(f_lm.landmark)
                smile_value = get_smile_ratio(f_lm.landmark)

                if not hands_up:
                    if mar_value > MAR_THRESHOLD:
                        face_state = "HABLANDO"
                    elif frown_value < FROWN_THRESHOLD:
                        face_state = "CENO FRUNCIDO" # Asociado a enojo/preocupación
                    elif smile_value > SMILE_THRESHOLD:
                        face_state = "SONRIENDO"     # Asociado a felicidad
                    else:
                        face_state = "NEUTRAL"

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, f_lm, mp_face_mesh.FACEMESH_TESSELATION,
                    None, mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())

        # 3. LÓGICA DE CONGRUENCIA (EL CEREBRO)
        # Regla 1: Texto Negativo pero Cara Sonriendo -> Incongruente
        if text_sentiment == "NEGATIVO" and face_state == "SONRIENDO":
            congruence_status = "INCONGRUENTE"
        # Regla 2: Texto Positivo pero Cara Enojada -> Incongruente
        elif text_sentiment == "POSITIVO" and face_state == "CENO FRUNCIDO":
            congruence_status = "INCONGRUENTE"
        else:
            congruence_status = "COHERENTE"

        # Guardar en CSV
        if writer:
            writer.writerow([datetime.now().strftime('%H:%M:%S'), face_state, text_sentiment, congruence_status, audio_sys.last_text])

        # 4. DIBUJAR DASHBOARD
        dashboard = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        dashboard[:] = (20, 20, 20) # Fondo casi negro

        # Video
        display_w = int(WINDOW_WIDTH * 0.65)
        display_h = int(display_w * (h_frame / w_frame))
        if display_h > WINDOW_HEIGHT: display_h = WINDOW_HEIGHT; display_w = int(display_h * (w_frame / h_frame))
        frame_resized = cv2.resize(frame, (display_w, display_h))
        dashboard[0:display_h, 0:display_w] = frame_resized

        # --- PANEL DE DATOS (DERECHA) ---
        x_start = display_w + 20
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Título Panel
        cv2.putText(dashboard, "ANALISIS REAL-TIME", (x_start, 40), font, 0.7, (150,150,150), 1)

        # A. GESTO
        cv2.putText(dashboard, "Gesto Detectado:", (x_start, 80), font, 0.6, (255,255,255), 1)
        cv2.putText(dashboard, face_state, (x_start, 110), font, 0.9, (0,255,255), 2)

        # B. SENTIMIENTO (TEXTO)
        cv2.putText(dashboard, "Sentimiento Discurso:", (x_start, 160), font, 0.6, (255,255,255), 1)
        color_sent = (200,200,200)
        if text_sentiment == "POSITIVO": color_sent = (0,255,0)
        elif text_sentiment == "NEGATIVO": color_sent = (0,0,255)
        cv2.putText(dashboard, text_sentiment, (x_start, 190), font, 0.9, color_sent, 2)

        # C. SEMÁFORO DE CONGRUENCIA
        cv2.rectangle(dashboard, (x_start, 240), (WINDOW_WIDTH - 20, 340), (50,50,50), -1) # Caja fondo

        if congruence_status == "COHERENTE":
            # Círculo Verde
            cv2.circle(dashboard, (x_start + 50, 290), 30, (0,255,0), -1)
            cv2.putText(dashboard, "COHERENTE", (x_start + 100, 300), font, 0.8, (0,255,0), 2)
        else:
            # Círculo Rojo parpadeante (simple)
            cv2.circle(dashboard, (x_start + 50, 290), 30, (0,0,255), -1)
            cv2.putText(dashboard, "ALERTA!", (x_start + 100, 300), font, 0.8, (0,0,255), 2)
            cv2.putText(dashboard, "Incongruencia", (x_start + 90, 325), font, 0.5, (200,200,200), 1)

        # 5. VENTANA TEXTO
        text_window = np.zeros((TEXT_W_HEIGHT, TEXT_W_WIDTH, 3), dtype=np.uint8)
        text_window[:] = (255, 255, 255)
        cv2.putText(text_window, "TRANSCRIPCION & ANALISIS", (20, 40), font, 0.8, (0, 0, 0), 2)
        draw_text_with_wrapping(text_window, audio_sys.last_text, font, 0.8, (50, 50, 50), 2, 20, 100, TEXT_W_WIDTH - 40)

        cv2.imshow('Analizador IA - Dashboard', dashboard)
        cv2.imshow('Transcripcion', text_window)

        if cv2.waitKey(5) & 0xFF == ord('q'): break

    audio_sys.stop()
    cap.release()
    if csv_file: csv_file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()