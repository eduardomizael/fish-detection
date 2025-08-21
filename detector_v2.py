from ultralytics import YOLO
import torch
import cv2
import numpy as np
import time
import os, tempfile, json
from textwrap import dedent
from collections import defaultdict


MODEL_FILE = r"runs\\detect\\train6\\weights\\best.pt"
VIDEO_CAPTURE_FILE = r'videos\\video04.mp4'

# =========================
# Vídeo
# =========================
# cap = cv2.VideoCapture(0)  # webcam
cap = cv2.VideoCapture(VIDEO_CAPTURE_FILE)  # arquivo
if not cap.isOpened():
    raise RuntimeError("Não foi possível abrir a fonte de vídeo.")

def loop_video(c):
    total = c.get(cv2.CAP_PROP_FRAME_COUNT)
    if total > 0 and c.get(cv2.CAP_PROP_POS_FRAMES) >= total - 1:
        c.set(cv2.CAP_PROP_POS_FRAMES, 0)

# =========================
# Tracker YAML (ByteTrack)
# =========================
tracker_yaml = dedent("""\
tracker_type: bytetrack
track_high_thresh: 0.6
new_track_thresh: 0.7
track_low_thresh: 0.1
match_thresh: 0.8
track_buffer: 60
fuse_score: True
mot20: False
""")
tmp_dir = tempfile.gettempdir()
tracker_path = os.path.join(tmp_dir, "custom_bytetrack.yaml")
with open(tracker_path, "w", encoding="utf-8") as f:
    f.write(tracker_yaml)

# =========================
# Modelo
# =========================
model = YOLO(MODEL_FILE)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE.startswith("cuda")
model.to(DEVICE)
IMG_SIZE = 480

# =========================
# Parâmetros anti-falso-positivo
# =========================
CONF_START    = 0.65     # conf p/ ENTRAR (promover ID)
DET_CONF_MIN  = 0.20     # conf mínima na detecção (baixa p/ tracker manter)
N_INIT        = 2        # frames bons seguidos para confirmar ID
LOST_PATIENCE = 45       # frames sem ver até "matar" ID ativo

# Filtro geométrico (opcional)
USE_GEOM_FILTER = False
MIN_AREA = 30 * 30
ASPECT_MIN, ASPECT_MAX = 1.2, 6.0

# ROI poligonal
USE_ROI = False
ROI_POINTS = None            # lista de (x,y)
ROI_MASK = None              # máscara 0/255
ROI_SAVE_PATH = os.path.join(tmp_dir, "roi_points.json")

# Estados de execução
seguir = True                # tracking on/off
deixar_rastro = True         # trilha on/off
_show_overlay = True         # overlay info (esquerda)
_show_shortcuts = True       # overlay atalhos (direita)

# Estados por ID
active_ids   = set()
last_seen    = defaultdict(int)
good_streak  = defaultdict(int)
track_history = defaultdict(list)
last_box     = {}

# =========================
# Funções de ROI
# =========================
def build_roi_mask(shape, points):
    if points is None or len(points) < 3:
        return None
    H, W = shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    poly = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [poly], 255)
    return mask

def apply_roi_mask(img, mask):
    if mask is None:
        return img
    return cv2.bitwise_and(img, img, mask=mask)

def save_roi(points, path=ROI_SAVE_PATH):
    if points is None or len(points) < 3:
        return False
    with open(path, "w", encoding="utf-8") as f:
        json.dump(points, f)
    return True

def load_roi(path=ROI_SAVE_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        pts = json.load(f)
    return [(int(x), int(y)) for x, y in pts] if pts else None

def edit_polygon_roi(frame, start_points=None):
    win = "Definir ROI (Clique pts, 'z' desfaz, 'r' reset, Enter OK, Esc/c cancela)"
    clone = frame.copy()
    pts = [] if start_points is None else list(start_points)

    def draw_preview(img):
        disp = img.copy()
        for p in pts:
            cv2.circle(disp, p, 4, (0, 255, 255), -1)
        if len(pts) >= 2:
            cv2.polylines(disp, [np.array(pts, dtype=np.int32)], False, (0, 255, 255), 2)
        cv2.putText(disp, "Clique pts | Enter=OK | z=Undo | r=Reset | Esc/c=Cancelar",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2, cv2.LINE_AA)
        return disp

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, max(640, frame.shape[1]//2), max(360, frame.shape[0]//2))
    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, draw_preview(clone))
        k = cv2.waitKey(10) & 0xFF
        if k in (13, 10):  # Enter
            if len(pts) >= 3:
                cv2.destroyWindow(win)
                return pts
        elif k in (27, ord('c')):  # Esc ou 'c'
            cv2.destroyWindow(win)
            return None
        elif k == ord('z'):
            if pts:
                pts.pop()
        elif k == ord('r'):
            pts.clear()

# =========================
# Filtro geométrico
# =========================
def geom_ok(xywh):
    if not USE_GEOM_FILTER:
        return True
    x, y, w, h = xywh
    area = float(w) * float(h)
    if area < MIN_AREA:
        return False
    ar1 = (w / (h + 1e-6))
    ar2 = (h / (w + 1e-6))
    ar = max(ar1, ar2)
    return ASPECT_MIN <= ar <= ASPECT_MAX

# =========================
# OVERLAYS (agrupados)
# =========================
def make_overlay_lines(fps, conf_start, n_init, patience, use_roi, geom_filter, tracking_on, trail_on, det_conf_min):
    return [
        f"FPS: {fps:.1f}",
        f"start>={conf_start:.2f}",
        f"Ninit: {n_init}",
        f"patience: {patience}f",
        f"ROI: {'on' if use_roi else 'off'}",
        f"geom: {'on' if geom_filter else 'off'}",
        f"track: {'on' if tracking_on else 'off'}",
        f"trail: {'on' if trail_on else 'off'}",
        f"det_conf>={det_conf_min:.2f}",
    ]

def make_shortcuts_lines():
    return [
        "ATALHOS:",
        "i  Mostrar/ocultar INFO",
        "h  Mostrar/ocultar ATALHOS",
        "q  Sair",
        "t  Tracking on/off",
        "r  Trilha on/off",
        "+/- Ajusta start conf",
        "9/0 Ajusta Ninit",
        "[/] Ajusta paciência",
        "a  Filtro geométrico on/off",
        "p  Editar ROI poligonal",
        "m  ROI on/off",
        "s  Salvar ROI",
        "l  Carregar ROI",
    ]

def draw_sidebar_panel(img, lines, side='left', enabled=True,
                       margin=10, x_pad=12, y_pad=12, line_gap=8,
                       font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, thickness=2,
                       text_color=(230, 230, 230), panel_color=(0, 0, 0), alpha=0.35):
    """
    Desenha um painel vertical semitransparente em 'left' ou 'right' com texto empilhado.
    """
    if not enabled or not lines:
        return img

    H, W = img.shape[:2]
    # Medidas de texto
    max_w = 0
    sizes = []
    for s in lines:
        (tw, th), _ = cv2.getTextSize(s, font, font_scale, thickness)
        max_w = max(max_w, tw)
        sizes.append((tw, th))

    panel_w = min(x_pad * 2 + max_w, W // 2)
    total_h = min(y_pad * 2 + sum(h for _, h in sizes) + line_gap * (len(lines) - 1), H - 20)

    # Posição
    y1 = margin
    y2 = y1 + total_h
    if side == 'left':
        x1 = margin
        x2 = x1 + panel_w
    else:  # 'right'
        x2 = W - margin
        x1 = x2 - panel_w

    # Fundo semitransparente
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), panel_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Texto
    cursor_y = y1 + y_pad
    for (s, (tw, th)) in zip(lines, sizes):
        cv2.putText(img, s, (x1 + x_pad, cursor_y + th), font, font_scale, text_color, thickness, cv2.LINE_AA)
        cursor_y += th + line_gap

    return img

# =========================
# Loop principal
# =========================
_prev_time = time.time()
_fps = 0.0

while True:
    success, frame = cap.read()
    if not success:
        loop_video(cap)
        continue
    loop_video(cap)

    # Atualiza paciência dos IDs ativos
    for tid in list(active_ids):
        last_seen[tid] += 1
        if last_seen[tid] > LOST_PATIENCE:
            active_ids.discard(tid)
            good_streak.pop(tid, None)
            track_history.pop(tid, None)
            last_box.pop(tid, None)

    # (Re)constrói máscara ROI se necessário
    if ROI_POINTS is not None and (ROI_MASK is None or ROI_MASK.shape[:2] != frame.shape[:2]):
        ROI_MASK = build_roi_mask(frame.shape, ROI_POINTS)

    # Aplica ROI
    img_infer = apply_roi_mask(frame, ROI_MASK) if (USE_ROI and ROI_MASK is not None) else frame

    # Inferência
    if seguir:
        results = model.track(
            img_infer,
            persist=True,
            conf=DET_CONF_MIN,
            iou=0.50,
            agnostic_nms=True,
            tracker=tracker_path,
            device=DEVICE,
            half=USE_HALF,
            imgsz=IMG_SIZE,
        )
    else:
        results = model(
            img_infer,
            conf=DET_CONF_MIN,
            iou=0.50,
            agnostic_nms=True,
            device=DEVICE,
            half=USE_HALF,
            imgsz=IMG_SIZE,
        )

    current_boxes = {}

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        xywh = boxes.xywh.cpu().numpy()
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones((len(xywh),), dtype=float)
        ids  = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xywh)

        for (b, c, tid) in zip(xywh, conf, ids):
            if tid is None:
                continue

            # confirmação temporal + filtro geométrico
            passed_filters = (c >= CONF_START) and geom_ok(b)
            good_streak[tid] = good_streak.get(tid, 0) + 1 if passed_filters else 0

            if (tid not in active_ids) and (good_streak[tid] >= N_INIT):
                active_ids.add(tid)
                last_seen[tid] = 0
                x, y, w, h = b
                track_history[tid] = [(float(x), float(y))]

            if tid in active_ids:
                last_seen[tid] = 0
                x, y, w, h = b
                hist = track_history[tid]
                hist.append((float(x), float(y)))
                if len(hist) > 30:
                    hist.pop(0)
                # box p/ desenhar
                x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
                current_boxes[tid] = (x1, y1, x2, y2, float(c))
                last_box[tid] = current_boxes[tid]

    # Desenho
    img_show = frame.copy()

    # Polígono ROI (se houver)
    if ROI_POINTS is not None:
        poly = np.array(ROI_POINTS, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_show, [poly], True, (255, 255, 0), 2)

    # Trilhas
    if deixar_rastro:
        for tid in active_ids:
            pts = track_history.get(tid, [])
            if len(pts) >= 2:
                points = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_show, [points], False, (0, 0, 230), 4)

    # Boxes apenas dos ativos no frame
    for tid, data in current_boxes.items():
        x1, y1, x2, y2, c = data
        cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img_show, f"id {tid}  {c:.2f}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA)

    # FPS para overlays
    now = time.time()
    dt = now - _prev_time
    _prev_time = now
    if dt > 0:
        _fps = 0.9 * _fps + 0.1 * (1.0 / dt) if _fps > 0 else (1.0 / dt)

    # -------- Overlays laterais --------
    if _show_overlay:
        lines_info = make_overlay_lines(
            fps=_fps, conf_start=CONF_START, n_init=N_INIT, patience=LOST_PATIENCE,
            use_roi=(USE_ROI and ROI_POINTS is not None),
            geom_filter=USE_GEOM_FILTER, tracking_on=seguir,
            trail_on=deixar_rastro, det_conf_min=DET_CONF_MIN
        )
        img_show = draw_sidebar_panel(img_show, lines_info, side='left', enabled=True)

    if _show_shortcuts:
        lines_help = make_shortcuts_lines()
        img_show = draw_sidebar_panel(img_show, lines_help, side='right', enabled=True)

    cv2.imshow("Tela", img_show)

    # Teclado
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('t'):
        seguir = not seguir
    elif k == ord('r'):
        deixar_rastro = not deixar_rastro
    elif k in (ord('+'), ord('=')):
        CONF_START = min(0.99, round(CONF_START + 0.05, 2))
    elif k in (ord('-'), ord('_')):
        CONF_START = max(0.05, round(CONF_START - 0.05, 2))
    elif k == ord(']'):
        LOST_PATIENCE = min(300, LOST_PATIENCE + 5)
    elif k == ord('['):
        LOST_PATIENCE = max(5, LOST_PATIENCE - 5)
    elif k == ord('0'):
        N_INIT = min(10, N_INIT + 1)
    elif k == ord('9'):
        N_INIT = max(1, N_INIT - 1)
    elif k == ord('a'):
        USE_GEOM_FILTER = not USE_GEOM_FILTER
    elif k == ord('p'):
        preview = frame.copy()
        pts = edit_polygon_roi(preview, ROI_POINTS)
        if pts is not None and len(pts) >= 3:
            ROI_POINTS = pts
            ROI_MASK = build_roi_mask(frame.shape, ROI_POINTS)
            USE_ROI = True
    elif k == ord('m'):
        if ROI_POINTS is not None:
            USE_ROI = not USE_ROI
    elif k == ord('s'):
        if ROI_POINTS is not None:
            save_roi(ROI_POINTS, ROI_SAVE_PATH)
    elif k == ord('l'):
        pts = load_roi(ROI_SAVE_PATH)
        if pts is not None and len(pts) >= 3:
            ROI_POINTS = pts
            ROI_MASK = build_roi_mask(frame.shape, ROI_POINTS)
            USE_ROI = True
    elif k == ord('i'):  # info esquerda
        _show_overlay = not _show_overlay
    elif k == ord('h'):  # atalhos direita
        _show_shortcuts = not _show_shortcuts

cap.release()
cv2.destroyAllWindows()
print("desligando")
