import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from scipy.signal import butter, filtfilt, detrend, welch
from scipy.ndimage import gaussian_filter1d
import time
import argparse
import serial

# ================================================================
# Settings
# ================================================================

# Forehead ROI (relative to inter-eye distance)
ROI_OFFSET_Y = 0.10  # higher on forehead
ROI_OFFSET_W = 0.60  # width
ROI_OFFSET_H = 0.40  # height

# HR band
HR_LOW_HZ  = 0.7    # 42 BPM
HR_HIGH_HZ = 2.0    # 120 BPM

# Motion rejection — nose displacement (px) to discard a frame
MOTION_THRESHOLD_PX = 8.0

# Sliding window — re-estimate HR every N clean frames
SLIDE_EVERY = 10

# Exposure lock
LOCK_EXPOSURE       = False
AUTO_EXPOSURE_VALUE = 0.25  # manual mode on most UVC drivers
EXPOSURE_VALUE      = -6    # log2 seconds; tune for your lighting

# Serial for HR output
ser = None
try: 
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=2)  # Adjust port and baud rate as needed
except Exception as e:
    print(f"Failed to open serial port: {e}")

# ================================================================
# Signal processing
# ================================================================

def bandpass_filter(signal, fs):
    nyq  = fs * 0.5
    b, a = butter(4, [HR_LOW_HZ / nyq, HR_HIGH_HZ / nyq], btype='band')
    return filtfilt(b, a, signal)

def normalize(sig):
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

def refine_peak(f, pxx, idx):
    """Parabolic interpolation for sub-bin frequency accuracy."""
    if idx <= 0 or idx >= len(pxx) - 1:
        return f[idx]
    a, b, c = pxx[idx - 1], pxx[idx], pxx[idx + 1]
    denom   = a - 2 * b + c
    if abs(denom) < 1e-10:
        return f[idx]
    return float(np.clip(f[idx] + 0.5 * (a - c) / denom * (f[1] - f[0]),
                         HR_LOW_HZ, HR_HIGH_HZ))

def estimate_hr(signal, fs):
    """Welch PSD with parabolic peak refinement. Always returns a BPM value."""
    f, pxx  = welch(signal, fs=fs, nperseg=max(len(signal) // 2, 64))
    mask    = (f >= HR_LOW_HZ) & (f <= HR_HIGH_HZ)
    f_sel, p_sel = f[mask], pxx[mask]
    idx     = int(np.argmax(p_sel))
    return refine_peak(f_sel, p_sel, idx) * 60.0

def process_signal(buffer, fs):
    sig = np.array(buffer, dtype=np.float32)
    sig = detrend(sig)
    sig = gaussian_filter1d(sig, sigma=1.5)
    sig = bandpass_filter(sig, fs)
    return normalize(sig)

# ================================================================
# rPPG methods
# ================================================================

def ir_method(roi):
    """Mean IR intensity after spatial smoothing."""
    gray = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    gray = gaussian_filter1d(gray, sigma=1, axis=0)
    gray = gaussian_filter1d(gray, sigma=1, axis=1)
    return float(np.mean(gray))

def chrom_method(roi):
    """CHROM projection with per-channel normalization."""
    R, G, B = (np.mean(roi[:, :, i]) for i in (2, 1, 0))
    total   = R + G + B + 1e-8
    Rn, Gn, Bn = R / total, G / total, B / total
    return (3 * Rn - 2 * Gn) - (1.5 * Rn + Gn - 1.5 * Bn)

# ================================================================
# ROI helpers
# ================================================================

def get_forehead_roi(frame, lm):
    h, w   = frame.shape[:2]
    c      = lm.landmark[10]
    cx, cy = int(c.x * w), int(c.y * h)
    L, R   = lm.landmark[33], lm.landmark[263]
    ed     = np.linalg.norm([(L.x - R.x) * w, (L.y - R.y) * h])
    rw, rh = int(ed * ROI_OFFSET_W), int(ed * ROI_OFFSET_H)
    fx     = max(0, min(cx - rw // 2,     w - rw))
    fy     = max(0, min(cy - int(ed * ROI_OFFSET_Y), h - rh))
    return fx, fy, rw, rh

def smooth_roi(prev, new, alpha=0.25, shape=None):
    """EMA-smooth ROI coords and clamp to frame bounds."""
    if prev is None:
        return new
    smoothed = tuple(int(p + alpha * (n - p)) for p, n in zip(prev, new))
    if shape is not None:
        fh, fw  = shape[:2]
        sx, sy, sw, sh = smoothed
        sx = max(0, min(sx, fw - sw))
        sy = max(0, min(sy, fh - sh))
        sw, sh = max(1, min(sw, fw)), max(1, min(sh, fh))
        return sx, sy, sw, sh
    return smoothed

# ================================================================
# Motion rejection
# ================================================================

def check_motion(lm, prev_nose, shape):
    """Return (displacement_px, should_reject) using nose-tip landmark."""
    h, w = shape[:2]
    nose = lm.landmark[1]
    pos  = (nose.x * w, nose.y * h)
    if prev_nose is None:
        return 0.0, False
    dist = float(np.linalg.norm([pos[0] - prev_nose[0], pos[1] - prev_nose[1]]))
    return dist, dist > MOTION_THRESHOLD_PX

# ================================================================
# FPS tracking
# ================================================================

_fps = {"count": 0, "t": None, "fps": 30.0}

def update_fps():
    now = time.time()
    if _fps["t"] is None:
        _fps["t"] = now
        return _fps["fps"]
    _fps["count"] += 1
    if (elapsed := now - _fps["t"]) >= 1.0:
        _fps["fps"]   = _fps["count"] / elapsed
        _fps["count"] = 0
        _fps["t"]     = now
    return _fps["fps"]

# ================================================================
# EMA smoother
# ================================================================

def smooth_value(prev, new, alpha=0.2):
    return new if prev is None else prev + alpha * (new - prev)

# ================================================================
# Exposure lock
# ================================================================

def lock_exposure(cap, value):
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_VALUE)
    cap.set(cv2.CAP_PROP_EXPOSURE, value)
    return cap.get(cv2.CAP_PROP_EXPOSURE)

# ================================================================
# CLI
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="IR/rPPG heart rate detector")
    p.add_argument("--gui",              action="store_true", help="Show live camera window")
    p.add_argument("--ir",            action="store_true", help="Use IR instead of CHROM method")
    p.add_argument("--buffer-seconds",   type=int,   default=10,                  help="Buffer length in seconds (default: 10)")
    p.add_argument("--slide-every",      type=int,   default=SLIDE_EVERY,         help="Re-estimate HR every N clean frames (default: %(default)s)")
    p.add_argument("--motion-thresh",    type=float, default=MOTION_THRESHOLD_PX, help="Nose displacement in px to reject frame (default: %(default)s)")
    p.add_argument("--no-lock-exposure", action="store_true",                     help="Skip exposure lock")
    p.add_argument("--exposure",         type=float, default=EXPOSURE_VALUE,      help="Manual exposure value (default: %(default)s)")
    p.add_argument("--roi-offset-y",     type=float, default=ROI_OFFSET_Y)
    p.add_argument("--roi-offset-w",     type=float, default=ROI_OFFSET_W)
    p.add_argument("--roi-offset-h",     type=float, default=ROI_OFFSET_H)
    return p.parse_args()

# ================================================================
# Serial Helper
# ================================================================

def send_hr_to_serial(hr):
    if ser is not None:
        try:
            ser.write(f"{hr:.1f}\n".encode('utf-8'))
            buf = ser.readline()  # Optional: read response from ESP32
            print(buf.decode('utf-8').strip())  # Optional: print response
        except Exception as e:
            print("Serial communication error")
            print(f"Error: {e}")
    else:
        print("Serial port not available. Cannot send HR data.")

# ================================================================
# Main
# ================================================================

def main(video_source=0):
    args = parse_args()

    global ROI_OFFSET_Y, ROI_OFFSET_W, ROI_OFFSET_H, MOTION_THRESHOLD_PX
    ROI_OFFSET_Y        = args.roi_offset_y
    ROI_OFFSET_W        = args.roi_offset_w
    ROI_OFFSET_H        = args.roi_offset_h
    MOTION_THRESHOLD_PX = args.motion_thresh

    prev_roi            = None
    prev_hr             = None
    prev_nose           = None
    last_hr_text        = None
    last_print_time     = 0.0
    frames_since_est    = 0
    fs                  = 30.0

    cap           = cv2.VideoCapture(video_source)
    signal_buffer = deque(maxlen=int(args.buffer_seconds * fs))

    if LOCK_EXPOSURE and not args.no_lock_exposure:
        actual = lock_exposure(cap, args.exposure)
        print(f"Exposure locked (requested {args.exposure}, camera reports {actual:.3f})")

    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as mesh:

        while True:
            success, frame = cap.read()
            if not success:
                break

            fs         = update_fps()
            target_len = int(args.buffer_seconds * fs)
            if abs(signal_buffer.maxlen - target_len) > 10:
                signal_buffer = deque(signal_buffer, maxlen=target_len)

            results = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_face_landmarks:
                lm             = results.multi_face_landmarks[0]
                motion_px, rejected = check_motion(lm, prev_nose, frame.shape)

                nose           = lm.landmark[1]
                h, w           = frame.shape[:2]
                prev_nose      = (nose.x * w, nose.y * h)

                prev_roi       = smooth_roi(prev_roi, get_forehead_roi(frame, lm), shape=frame.shape)
                fx, fy, fw, fh = prev_roi

                if args.gui:
                    color = (0, 255, 255) if rejected else (255, 0, 0)
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)

                if not rejected:
                    patch = frame[fy:fy + fh, fx:fx + fw]
                    if patch.size > 0:
                        sig_val = ir_method(patch) if args.ir else chrom_method(patch)
                        signal_buffer.append(sig_val)
                        frames_since_est += 1

                if len(signal_buffer) == signal_buffer.maxlen and frames_since_est >= args.slide_every:
                    frames_since_est = 0
                    hr               = estimate_hr(process_signal(signal_buffer, fs), fs)
                    prev_hr          = smooth_value(prev_hr, hr)
                    last_hr_text     = f"Heart Rate: {prev_hr:.1f} BPM"

                    # Send HR to ESP32 via Serial
                    now = time.time()
                    if now - last_print_time >= 1.0 and prev_hr is not None:
                        print(last_hr_text)
                        send_hr_to_serial(prev_hr)
                        last_print_time = now

                if args.gui:
                    if last_hr_text:
                        cv2.putText(frame, last_hr_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if rejected:
                        cv2.putText(frame, f"Motion! ({motion_px:.1f}px) — frame skipped",
                                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                elif rejected:
                    print(f"Motion detected ({motion_px:.1f}px) — frame skipped")

            else:
                prev_nose = None
                if args.gui:
                    cv2.putText(frame, "No face detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                now = time.time()
                if now - last_print_time >= 1.0:
                    print("No face detected")
                    last_print_time = now

            if args.gui:
                cv2.imshow("Heart Rate Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    if ser is not None:
        ser.close()
    if args.gui:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()