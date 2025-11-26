import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from scipy.signal import butter, filtfilt, detrend, welch
from scipy.ndimage import gaussian_filter1d
import time
import serial
import struct

# ================================================================
# Adjustable settings
# ================================================================
SERIAL_PORT = 'COM5'

mp_face_mesh = mp.solutions.face_mesh

# ================================================================
# Filtering functions
# ================================================================
def bandpass_filter(signal, fs, lowcut=0.7, highcut=2.0, order=4):
    nyq = fs * 0.5
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

def moving_average(signal, window=5):
    return np.convolve(signal, np.ones(window)/window, mode='same')

def normalize(sig):
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

# ================================================================
# Peak refinement 
# ================================================================
def refine_peak(f, pxx, idx):
    if idx <= 0 or idx >= len(pxx)-1:
        return f[idx]
    alpha = pxx[idx - 1]
    beta  = pxx[idx]
    gamma = pxx[idx + 1]

    p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
    return f[idx] + p * (f[1] - f[0])

# ================================================================
# CHROM rPPG method 
# ================================================================
def chrom_method(roi):
    R = np.mean(roi[:,:,2])
    G = np.mean(roi[:,:,1])
    B = np.mean(roi[:,:,0])
    X = 3*R - 2*G
    Y = 1.5*R + G - 1.5*B
    return X - Y

# ================================================================
# Adaptive ROI smoother
# ================================================================
def smooth_roi(prev, new, alpha=0.25):
    if prev is None:
        return new
    px, py, pw, ph = prev
    nx, ny, nw, nh = new
    sx = int(px + alpha*(nx - px))
    sy = int(py + alpha*(ny - py))
    sw = int(pw + alpha*(nw - pw))
    sh = int(ph + alpha*(nh - ph))
    return (sx, sy, sw, sh)

# ================================================================
# Forehead ROI — small, stable, very accurate
# ================================================================
def get_forehead_roi(frame, lm):
    h, w, _ = frame.shape

    # forehead-center landmark
    center = lm.landmark[10]
    cx = int(center.x * w)
    cy = int(center.y * h)

    # Eye landmarks for scale
    L = lm.landmark[33]     # Right eye
    R = lm.landmark[263]    # Left eye
    eye_dist = np.linalg.norm([
        (L.x - R.x) * w,
        (L.y - R.y) * h
    ])

    # ROI sizing (tuned)
    roi_w = int(eye_dist * 0.60) # Higher means wider
    roi_h = int(eye_dist * 0.40) # Higher means taller
    offset_y = int(eye_dist * 0.10) # Higher means higher relative to eyes

    fx = cx - roi_w//2
    fy = cy - offset_y

    # Clamp to screen
    fx = max(0, min(fx, w - roi_w))
    fy = max(0, min(fy, h - roi_h))

    return (fx, fy, roi_w, roi_h)

# ================================================================
# FPS measurement to avoid heart-rate inflation
# ================================================================
frame_counter = 0
last_fps_time = time.time()
measured_fps = 30.0

def update_fps():
    global frame_counter, last_fps_time, measured_fps
    frame_counter += 1
    now = time.time()
    if now - last_fps_time >= 1.0:
        measured_fps = frame_counter / (now - last_fps_time)
        frame_counter = 0
        last_fps_time = now
    return measured_fps

# ================================================================
# HR estimation
# ================================================================
def estimate_hr(signal, fs):
    f, pxx = welch(signal, fs=fs, nperseg=len(signal))

    # bandmask already filtered, but double-check
    mask = (f >= 0.7) & (f <= 2.0)
    f_sel = f[mask]
    p_sel = pxx[mask]

    idx = np.argmax(p_sel)
    refined = refine_peak(f_sel, p_sel, idx)
    bpm = refined * 60.0
    return bpm

# EMA smoother
def smooth_value(prev, new, alpha=0.2):
    if prev is None:
        return new
    return prev + alpha*(new - prev)

# ================================================================
# Main Program
# ================================================================
def main(video_source=0):
    try:
        # Open serial port
        ser = serial.Serial(SERIAL_PORT, baudrate = 9600, timeout = 1)
        print(f"Serial port {ser.name} opened successfully!")

        cap = cv2.VideoCapture(video_source)

        buffer_seconds = 20          # 20 seconds = ~3 BPM resolution
        prev_roi = None
        prev_hr = None

        # Initial guess
        fs = 30.0

        signal_buffer = deque(maxlen=int(buffer_seconds * fs))

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as mesh:

            while True:
                success, frame = cap.read()
                if not success:
                    break

                fs = update_fps()  # REAL FPS
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mesh.process(gray)

                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0]

                    # ROI detection
                    new_roi = get_forehead_roi(frame, lm)
                    roi = smooth_roi(prev_roi, new_roi)
                    prev_roi = roi

                    fx, fy, fw, fh = roi
                    cv2.rectangle(frame, (fx,fy), (fx+fw, fy+fh), (255,0,0), 2)

                    patch = frame[fy:fy+fh, fx:fx+fw]

                    if patch.size > 0:
                        sig_val = chrom_method(patch)
                        signal_buffer.append(sig_val)

                    # Only estimate HR once buffer filled
                    if len(signal_buffer) == signal_buffer.maxlen:
                        sig = np.array(signal_buffer, dtype=np.float32)

                        # Full filter chain
                        # Denoise > normalize > detrend > moving average > fft
                        sig = gaussian_filter1d(sig, sigma=1.5)
                        sig = normalize(sig)
                        sig = detrend(sig)
                        sig = moving_average(sig, window=5)
                        sig = bandpass_filter(sig, fs)

                        hr = estimate_hr(sig, fs)

                        # Smooth HR value for less jitter
                        hr_sm = smooth_value(prev_hr, hr, alpha=0.2)
                        prev_hr = hr_sm

                        cv2.putText(frame,
                            f"HR: {hr_sm:.1f} BPM",
                            (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,0,255),
                            2)
                        
                        # Send HR over serial as a float
                        data = struct.pack('<f', hr_sm)
                        ser.write(data)

                cv2.imshow("rPPG (High Accuracy Version)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    except serial.SerialException as e:
        print(f"Error: {e}")

    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed!")


if __name__ == "__main__":
    main()
