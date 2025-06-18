import cv2
import numpy as np
import torch
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData

# Allow loading scaler
import torch.serialization
torch.serialization.add_safe_globals([StandardScaler])

checkpoint = torch.load("gesture_model.pt", map_location="cpu", weights_only=False)
def clean_sd(sd): return {k[len("model.") or 0:]: v for k,v in sd.items()}
model = torch.nn.Sequential(
    torch.nn.Linear(63, 128), torch.nn.ReLU(),
    torch.nn.Linear(128,64), torch.nn.ReLU(),
    torch.nn.Linear(64,5)
)
model.load_state_dict(clean_sd(checkpoint["model_state_dict"]))
model.eval()
scaler = checkpoint["scaler"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def extract_features(hand_landmarks):
    return np.array([[lm.x for lm in hand_landmarks.landmark] +
                     [lm.y for lm in hand_landmarks.landmark] +
                     [lm.z for lm in hand_landmarks.landmark]], dtype=np.float32)

def gesture_to_controls(gid):
    # Map gestures to two joint angles
    mapping = {
        0: [0.0, 0.0],        # Fist → rest
        1: [0.2, -0.2],       # Open palm → some pose
        2: [-0.2, 0.2],       # Left → reverse
        3: [0.2, 0.0],        # Right → start
        4: [0.0, 0.5],        # Thumbs up → lift elbow
    }
    return mapping.get(gid, [0.0, 0.0])

mj_model = MjModel.from_xml_path("gesture_arm.xml")
mj_data = MjData(mj_model)

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    cap = cv2.VideoCapture(0)
    while viewer.is_running():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        gid = None
        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                feats = extract_features(lm)
                norm = scaler.transform(feats)
                pred = model(torch.from_numpy(norm).float())
                gid = torch.argmax(pred, dim=1).item()
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
        ctrl_vals = gesture_to_controls(gid) if gid is not None else [0,0]
        mj_data.ctrl[mj_model.actuator("shoulder").id] = ctrl_vals[0]
        mj_data.ctrl[mj_model.actuator("elbow").id] = ctrl_vals[1]
        mujoco.mj_step(mj_model, mj_data)
        viewer.label = f"Gesture: {gid}"
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()
