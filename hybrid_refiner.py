import os
import json
import math
import uuid
import asyncio
import ctypes
from ctypes import wintypes
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import PIL.Image
import PIL.ImageOps

import folder_paths
import comfy.utils
import comfy.samplers
import comfy.sample
import comfy.model_management
import nodes

from .InstantID import draw_kps, _set_model_patch_replace
from .utils import tensor_to_image, image_to_tensor

# Cache in RAM for folder reference embeddings & keypoints so repeated generations are instant
_REF_FOLDER_CACHE = {}

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

USER_PRESETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_presets.json")

BUILTIN_PRESETS = {
    "manual (usar controles abaixo)": {},
    "default (padrão equilibrado)": {
        "ip_weight": 0.95,
        "cn_strength": 0.85,
        "kps_identity_morph": 0.65,
        "crop_scale": 2.20,
        "denoise": 0.55,
        "steps": 12,
        "cfg": 1.0,
        "mask_expand": 16,
        "mask_blur": 32,
        "mouth_mask_keep": 0.0,
        "nose_mask_keep": 0.0,
        "hf_pore_transfer": 0.45,
    },
    "high_fidelity (alta fidelidade facial sem distorções)": {
        "ip_weight": 1.08,
        "cn_strength": 0.85,
        "kps_identity_morph": 0.90,
        "crop_scale": 1.95,
        "denoise": 0.64,
        "steps": 15,
        "cfg": 1.0,
        "mask_expand": 18,
        "mask_blur": 32,
        "mouth_mask_keep": 0.0,
        "nose_mask_keep": 0.0,
        "hf_pore_transfer": 0.30,
    },
    "strong_expression (expressão forte / língua para fora - sem máscara)": {
        "ip_weight": 1.12,
        "cn_strength": 0.60,
        "kps_identity_morph": 0.20,
        "crop_scale": 2.10,
        "denoise": 0.43,
        "steps": 14,
        "cfg": 1.0,
        "mask_expand": 16,
        "mask_blur": 32,
        "mouth_mask_keep": 0.0,
        "nose_mask_keep": 0.0,
        "hf_pore_transfer": 0.45,
    },
    "mouth_object (cigarro / canudo / objeto na boca - sem máscara)": {
        "ip_weight": 1.15,
        "cn_strength": 0.55,
        "kps_identity_morph": 0.15,
        "crop_scale": 2.15,
        "denoise": 0.38,
        "steps": 14,
        "cfg": 1.0,
        "mask_expand": 16,
        "mask_blur": 32,
        "mouth_mask_keep": 0.0,
        "nose_mask_keep": 0.0,
        "hf_pore_transfer": 0.50,
    },
    "piercing_detail (piercing no nariz / sobrancelha - sem máscara)": {
        "ip_weight": 1.10,
        "cn_strength": 0.75,
        "kps_identity_morph": 0.65,
        "crop_scale": 2.05,
        "denoise": 0.42,
        "steps": 14,
        "cfg": 1.0,
        "mask_expand": 16,
        "mask_blur": 32,
        "mouth_mask_keep": 0.0,
        "nose_mask_keep": 0.0,
        "hf_pore_transfer": 0.60,
    },
    "mask_keep_mouth (alta fidelidade + proteger boca/língua/cigarro na máscara)": {
        "ip_weight": 1.08,
        "cn_strength": 0.80,
        "kps_identity_morph": 0.85,
        "crop_scale": 2.00,
        "denoise": 0.62,
        "steps": 15,
        "cfg": 1.0,
        "mask_expand": 18,
        "mask_blur": 32,
        "mouth_mask_keep": 0.85,
        "nose_mask_keep": 0.0,
        "hf_pore_transfer": 0.40,
    },
    "mask_keep_nose_mouth (alta fidelidade + proteger boca e piercing no nariz na máscara)": {
        "ip_weight": 1.08,
        "cn_strength": 0.80,
        "kps_identity_morph": 0.85,
        "crop_scale": 2.00,
        "denoise": 0.62,
        "steps": 15,
        "cfg": 1.0,
        "mask_expand": 18,
        "mask_blur": 32,
        "mouth_mask_keep": 0.85,
        "nose_mask_keep": 0.85,
        "hf_pore_transfer": 0.45,
    },
}


class DynamicPresetList(list):
    """
    A list subclass for ComfyUI COMBO widgets that allows dynamically added user presets
    to pass ComfyUI's `x in combo_list` input validation without restarting the server.
    """

    def __contains__(self, item):
        return True


def load_user_presets():
    if os.path.isfile(USER_PRESETS_PATH):
        try:
            with open(USER_PRESETS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            print(f"[InstantID Hybrid] Aviso ao ler user_presets.json: {e}")
    return {}


def save_user_presets(user_presets_dict):
    try:
        with open(USER_PRESETS_PATH, "w", encoding="utf-8") as f:
            json.dump(user_presets_dict, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[InstantID Hybrid] Erro ao salvar user_presets.json: {e}")
        return False


def get_all_presets():
    merged = dict(BUILTIN_PRESETS)
    user_p = load_user_presets()
    for k, v in user_p.items():
        if isinstance(v, dict):
            merged[k] = v
    return merged


def get_preset_options_list():
    return DynamicPresetList(list(get_all_presets().keys()))



# ==============================================================================
# Native Windows Explorer Folder Picker (IFileOpenDialog via ctypes) + API Route
# ==============================================================================
def open_windows_folder_dialog(initial_dir="", title="Selecione a Pasta de Fotos de Referência (InstantID)"):
    """
    Opens the modern Windows 10/11 Explorer Folder Picker dialog (IFileOpenDialog with FOS_PICKFOLDERS)
    directly in front of the active browser window using pure ctypes (zero external dependencies).
    Returns the selected folder path string or None if cancelled.
    """
    if os.name != "nt":
        return None

    ole32 = ctypes.windll.ole32
    shell32 = ctypes.windll.shell32
    user32 = ctypes.windll.user32

    # Initialize COM in STA mode on this worker thread
    ole32.CoInitializeEx(None, 2)  # COINIT_APARTMENTTHREADED = 2

    clsid_fod = (ctypes.c_byte * 16).from_buffer_copy(
        uuid.UUID("{DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7}").bytes_le
    )
    iid_fod = (ctypes.c_byte * 16).from_buffer_copy(
        uuid.UUID("{d57c7288-d4ad-4768-be02-9d969532d960}").bytes_le
    )
    iid_si = (ctypes.c_byte * 16).from_buffer_copy(
        uuid.UUID("{43826d1e-e718-42ee-bc55-a1e261c37bfe}").bytes_le
    )

    p_dlg = ctypes.c_void_p()
    hr = ole32.CoCreateInstance(
        ctypes.byref(clsid_fod),
        None,
        1,  # CLSCTX_INPROC_SERVER
        ctypes.byref(iid_fod),
        ctypes.byref(p_dlg),
    )
    if hr != 0 or not p_dlg.value:
        ole32.CoUninitialize()
        return None

    selected_path = None
    try:
        vtbl = ctypes.cast(
            ctypes.cast(p_dlg, ctypes.POINTER(ctypes.c_void_p))[0],
            ctypes.POINTER(ctypes.c_void_p),
        )

        Release = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(vtbl[2])
        Show = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.HWND)(vtbl[3])
        SetOptions = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.DWORD)(vtbl[9])
        SetFolder = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(vtbl[12])
        SetTitle = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.LPCWSTR)(vtbl[17])
        GetResult = ctypes.WINFUNCTYPE(
            ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )(vtbl[20])

        # FOS_PICKFOLDERS (0x20) | FOS_FORCEFILESYSTEM (0x40) | FOS_PATHMUSTEXIST (0x800)
        SetOptions(p_dlg, 0x20 | 0x40 | 0x800)
        SetTitle(p_dlg, title)

        # Resolve initial directory (fallback to parent directory if needed)
        init_clean = (initial_dir or "").strip().strip('"').strip("'")
        if init_clean and not os.path.isdir(init_clean):
            parent = os.path.dirname(init_clean)
            if os.path.isdir(parent):
                init_clean = parent
            else:
                init_clean = ""

        if init_clean and os.path.isdir(init_clean):
            p_init_si = ctypes.c_void_p()
            hr_si = shell32.SHCreateItemFromParsingName(
                wintypes.LPCWSTR(os.path.abspath(init_clean)),
                None,
                ctypes.byref(iid_si),
                ctypes.byref(p_init_si),
            )
            if hr_si == 0 and p_init_si.value:
                SetFolder(p_dlg, p_init_si)
                si_vtbl = ctypes.cast(
                    ctypes.cast(p_init_si, ctypes.POINTER(ctypes.c_void_p))[0],
                    ctypes.POINTER(ctypes.c_void_p),
                )
                si_release = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(si_vtbl[2])
                si_release(p_init_si)

        hwnd_owner = user32.GetForegroundWindow()
        hr_show = Show(p_dlg, hwnd_owner)
        if hr_show == 0:
            p_res_si = ctypes.c_void_p()
            if GetResult(p_dlg, ctypes.byref(p_res_si)) == 0 and p_res_si.value:
                res_vtbl = ctypes.cast(
                    ctypes.cast(p_res_si, ctypes.POINTER(ctypes.c_void_p))[0],
                    ctypes.POINTER(ctypes.c_void_p),
                )
                GetDisplayName = ctypes.WINFUNCTYPE(
                    ctypes.c_long,
                    ctypes.c_void_p,
                    wintypes.DWORD,
                    ctypes.POINTER(wintypes.LPWSTR),
                )(res_vtbl[5])
                ResRelease = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(res_vtbl[2])

                psz_path = wintypes.LPWSTR()
                # SIGDN_FILESYSPATH = 0x80058000
                if GetDisplayName(p_res_si, 0x80058000, ctypes.byref(psz_path)) == 0 and psz_path.value:
                    selected_path = str(psz_path.value)
                    ole32.CoTaskMemFree(psz_path)
                ResRelease(p_res_si)

        Release(p_dlg)
    finally:
        ole32.CoUninitialize()

    return selected_path


try:
    import server
    from aiohttp import web

    @server.PromptServer.instance.routes.post("/instantid_hybrid/select_folder")
    async def api_select_reference_folder(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        current_path = data.get("current_path", "")
        title = data.get("title", "Selecione a Pasta de Fotos de Referência (InstantID)")

        selected = await asyncio.to_thread(open_windows_folder_dialog, current_path, title)
        if not selected:
            return web.json_response({"status": "cancelled"})

        img_count = 0
        if os.path.isdir(selected):
            img_count = sum(
                1
                for f in os.listdir(selected)
                if os.path.splitext(f)[1].lower() in VALID_IMAGE_EXTS
            )
        return web.json_response(
            {
                "status": "ok",
                "path": selected,
                "image_count": img_count,
            }
        )

    @server.PromptServer.instance.routes.get("/instantid_hybrid/presets")
    async def api_get_presets(request):
        return web.json_response(
            {
                "status": "ok",
                "presets": get_all_presets(),
                "builtin_names": list(BUILTIN_PRESETS.keys()),
            }
        )

    @server.PromptServer.instance.routes.post("/instantid_hybrid/save_preset")
    async def api_save_preset(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"status": "error", "message": "Invalid JSON"}, status=400)

        name = (data.get("name") or "").strip()
        values = data.get("values") or {}
        if not name or name in BUILTIN_PRESETS:
            return web.json_response(
                {"status": "error", "message": "Escolha um nome personalizado diferente dos presets padrão."},
                status=400,
            )

        user_p = load_user_presets()
        user_p[name] = values
        save_user_presets(user_p)
        return web.json_response(
            {
                "status": "ok",
                "name": name,
                "presets": get_all_presets(),
                "builtin_names": list(BUILTIN_PRESETS.keys()),
            }
        )

    @server.PromptServer.instance.routes.post("/instantid_hybrid/delete_preset")
    async def api_delete_preset(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"status": "error", "message": "Invalid JSON"}, status=400)

        name = (data.get("name") or "").strip()
        user_p = load_user_presets()
        if name in user_p:
            del user_p[name]
            save_user_presets(user_p)
        return web.json_response(
            {
                "status": "ok",
                "presets": get_all_presets(),
                "builtin_names": list(BUILTIN_PRESETS.keys()),
            }
        )
except Exception:
    pass


# ==============================================================================
# Face Detection, Gender Filtering & Spatial Position Selection
# ==============================================================================
def detect_best_face(insightface, img_bgr):
    """
    Detects faces in a BGR uint8 numpy image (H, W, 3) with automatic fallback resolutions.
    Returns the largest detected face dict or None if no face is found.
    """
    insightface.det_model.input_size = (640, 640)
    for size in range(640, 128, -64):
        insightface.det_model.input_size = (size, size)
        faces = insightface.get(img_bgr)
        if faces:
            best_face = sorted(
                faces,
                key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
            )[-1]
            insightface.det_model.input_size = (640, 640)
            return best_face
    insightface.det_model.input_size = (640, 640)
    return None


def detect_all_faces(insightface, img_bgr, min_score=0.35):
    """
    Detects ALL valid faces in a BGR uint8 numpy image (H, W, 3).
    Filters out tiny background false positives.
    """
    h, w, _ = img_bgr.shape
    min_dim = min(h, w)
    insightface.det_model.input_size = (640, 640)
    for size in range(640, 128, -64):
        insightface.det_model.input_size = (size, size)
        faces = insightface.get(img_bgr)
        if faces:
            valid = []
            for f in faces:
                score = float(f.get("det_score", 1.0))
                x1, y1, x2, y2 = [float(v) for v in f["bbox"]]
                bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
                if score >= min_score and bw >= max(20.0, min_dim * 0.018) and bh >= max(20.0, min_dim * 0.018):
                    valid.append(f)
            if valid:
                insightface.det_model.input_size = (640, 640)
                return valid
    insightface.det_model.input_size = (640, 640)
    return []


def _face_cx(f):
    return 0.5 * (float(f["bbox"][0]) + float(f["bbox"][2]))


def _face_cy(f):
    return 0.5 * (float(f["bbox"][1]) + float(f["bbox"][3]))


def _face_area(f):
    return max(0.0, float(f["bbox"][2]) - float(f["bbox"][0])) * max(
        0.0, float(f["bbox"][3]) - float(f["bbox"][1])
    )


def sort_and_pick_faces(cand_faces, img_w, img_h, target_position, target_index):
    """
    Orders candidate faces according to `target_position` and picks either all faces
    (when target_position starts with 'all') or the 1-based `target_index` face.
    """
    if not cand_faces:
        return []

    if target_position.startswith("all"):
        # Sort left-to-right for deterministic multi-face refinement
        return sorted(cand_faces, key=_face_cx)

    if target_position.startswith("largest"):
        ordered = sorted(cand_faces, key=lambda f: -_face_area(f))
    elif target_position.startswith("left_to_right"):
        ordered = sorted(cand_faces, key=_face_cx)
    elif target_position.startswith("right_to_left"):
        ordered = sorted(cand_faces, key=lambda f: -_face_cx(f))
    elif target_position.startswith("top_to_bottom"):
        ordered = sorted(cand_faces, key=_face_cy)
    elif target_position.startswith("bottom_to_top"):
        ordered = sorted(cand_faces, key=lambda f: -_face_cy(f))
    elif target_position.startswith("center"):
        if len(cand_faces) >= 3:
            med_x = float(np.median([_face_cx(f) for f in cand_faces]))
            med_y = float(np.median([_face_cy(f) for f in cand_faces]))
            ordered = sorted(
                cand_faces,
                key=lambda f: abs(_face_cx(f) - med_x) + 0.25 * abs(_face_cy(f) - med_y),
            )
        else:
            cx_img, cy_img = 0.5 * float(img_w), 0.5 * float(img_h)
            ordered = sorted(
                cand_faces,
                key=lambda f: (_face_cx(f) - cx_img) ** 2 + (_face_cy(f) - cy_img) ** 2,
            )
    else:
        ordered = sorted(cand_faces, key=lambda f: -_face_area(f))

    idx = max(0, min(int(target_index) - 1, len(ordered) - 1))
    return [ordered[idx]]


def _is_female(f):
    g = f.get("gender", getattr(f, "gender", None))
    if g is None:
        s = getattr(f, "sex", None)
        return str(s).upper() == "F"
    if isinstance(g, str):
        return g.upper() in ("F", "0", "FEMALE")
    return int(g) == 0


def _is_male(f):
    g = f.get("gender", getattr(f, "gender", None))
    if g is None:
        s = getattr(f, "sex", None)
        return str(s).upper() == "M"
    if isinstance(g, str):
        return g.upper() in ("M", "1", "MALE")
    return int(g) == 1


def select_target_faces(all_faces, img_w, img_h, target_gender, target_position, target_index):
    """
    Filters and selects target faces from `all_faces` based on `target_gender`, `target_position`,
    and 1-based `target_index`.
    Returns a list of tuples: [(face_dict, ref_slot), ...] where ref_slot is 1 (primary reference)
    or 2 (secondary reference used for the male partner in couple_both mode).
    """
    if not all_faces:
        return []

    # Couple mode: ref_slot=1 for female, ref_slot=2 for male
    if target_gender.startswith("couple_both"):
        females = [f for f in all_faces if _is_female(f)]
        males = [f for f in all_faces if _is_male(f)]

        # Smart fallback if InsightFace misclassified gender on one partner in a 2+ person scene
        if len(all_faces) >= 2 and (not females or not males):
            top2 = sorted(all_faces, key=lambda f: -_face_area(f))[:2]
            # Typically the female face in a couple has a slightly smaller bbox or lower cy
            top2_sorted = sorted(top2, key=_face_area)
            females = [top2_sorted[0]]
            males = [top2_sorted[1]]
            print(
                "\033[33m[InstantID Hybrid] Info: Fallback inteligente de casal ativado (2 faces separadas para Ref 1 e Ref 2).\033[0m"
            )

        picked_f = sort_and_pick_faces(females, img_w, img_h, target_position, target_index)
        picked_m = sort_and_pick_faces(males, img_w, img_h, target_position, target_index)
        return [(f, 1) for f in picked_f] + [(m, 2) for m in picked_m]

    elif target_gender.startswith("female_only"):
        cand = [f for f in all_faces if _is_female(f)]
        if not cand and len(all_faces) == 1:
            print(
                "\033[33m[InstantID Hybrid] Aviso: Nenhuma face classificada como feminina (gender=0); usando a única face detectada.\033[0m"
            )
            cand = all_faces
        picked = sort_and_pick_faces(cand, img_w, img_h, target_position, target_index)
        return [(f, 1) for f in picked]

    elif target_gender.startswith("male_only"):
        cand = [f for f in all_faces if _is_male(f)]
        if not cand and len(all_faces) == 1:
            print(
                "\033[33m[InstantID Hybrid] Aviso: Nenhuma face classificada como masculina (gender=1); usando a única face detectada.\033[0m"
            )
            cand = all_faces
        picked = sort_and_pick_faces(cand, img_w, img_h, target_position, target_index)
        return [(f, 1) for f in picked]

    else:  # all (todos)
        picked = sort_and_pick_faces(all_faces, img_w, img_h, target_position, target_index)
        return [(f, 1) for f in picked]


# ==============================================================================
# Cranial Keypoint Morphing, 106-Landmark Masking & Color/Texture Matching
# ==============================================================================
def compute_face_frontalness(kps):
    """
    Computes a frontalness score in [0.0, 1.0] from 5 facial keypoints:
    0: left_eye, 1: right_eye, 2: nose, 3: mouth_left, 4: mouth_right.
    """
    kps = np.asarray(kps, dtype=np.float32)
    left_eye, right_eye, nose, mouth_l, mouth_r = kps[0], kps[1], kps[2], kps[3], kps[4]
    eye_vec = right_eye - left_eye
    eye_dist_sq = float(np.dot(eye_vec, eye_vec))
    if eye_dist_sq < 1e-4:
        return 0.0

    t_nose = float(np.dot(nose - left_eye, eye_vec) / eye_dist_sq)
    mouth_mid = 0.5 * (mouth_l + mouth_r)
    t_mouth = float(np.dot(mouth_mid - left_eye, eye_vec) / eye_dist_sq)

    yaw_sym_nose = max(0.0, 1.0 - 2.4 * abs(t_nose - 0.5))
    yaw_sym_mouth = max(0.0, 1.0 - 2.4 * abs(t_mouth - 0.5))
    return float(np.clip(yaw_sym_nose * yaw_sym_mouth, 0.0, 1.0))


def extract_reference_data(insightface, use_reference_folder, reference_folder, reference_image=None):
    """
    Extracts (embeddings, kps_list, weights_list) from either a folder of images (cached)
    or an input IMAGE tensor batch.
    """
    folder_clean = (reference_folder or "").strip().strip('"').strip("'")
    folder_valid = bool(folder_clean and os.path.isdir(folder_clean))

    use_folder = use_reference_folder and folder_valid
    if not use_folder and reference_image is None and folder_valid:
        use_folder = True

    if use_folder:
        files = sorted(
            [
                os.path.join(folder_clean, f)
                for f in os.listdir(folder_clean)
                if os.path.splitext(f)[1].lower() in VALID_IMAGE_EXTS
            ]
        )
        if not files:
            raise ValueError(f"Nenhuma imagem encontrada na pasta de referência: {folder_clean}")

        max_mtime = max(os.path.getmtime(f) for f in files)
        cache_key = (os.path.abspath(folder_clean), len(files), round(max_mtime, 2))

        if cache_key in _REF_FOLDER_CACHE:
            return _REF_FOLDER_CACHE[cache_key]

        embeds = []
        kps_list = []
        scores = []

        for fpath in files:
            try:
                pil_img = PIL.Image.open(fpath)
                pil_img = PIL.ImageOps.exif_transpose(pil_img).convert("RGB")
                if max(pil_img.size) > 1536:
                    pil_img.thumbnail((1536, 1536), PIL.Image.Resampling.LANCZOS)
                img_rgb = np.array(pil_img, dtype=np.uint8)
                img_bgr = img_rgb[:, :, ::-1].copy()
                face = detect_best_face(insightface, img_bgr)
                if face is not None and "embedding" in face:
                    embeds.append(torch.from_numpy(face["embedding"]).float())
                    kps_list.append(np.array(face["kps"], dtype=np.float32))
                    scores.append(float(face.get("det_score", 0.9)))
            except Exception as e:
                print(f"[InstantID Hybrid] Ignorando arquivo {fpath}: {e}")

        if not embeds:
            raise ValueError(f"Nenhuma face detectada nas imagens da pasta: {folder_clean}")

        result = (embeds, kps_list, scores)
        _REF_FOLDER_CACHE[cache_key] = result
        print(f"[InstantID Hybrid] {len(embeds)} faces de referência carregadas e cacheadas de: {folder_clean}")
        return result

    if reference_image is None:
        raise ValueError(
            "Forneça uma 'reference_image' ou ative 'use_reference_folder' com um caminho de pasta válido."
        )

    face_imgs_bgr = tensor_to_image(reference_image)
    embeds = []
    kps_list = []
    scores = []

    for i in range(face_imgs_bgr.shape[0]):
        face = detect_best_face(insightface, face_imgs_bgr[i])
        if face is not None and "embedding" in face:
            embeds.append(torch.from_numpy(face["embedding"]).float())
            kps_list.append(np.array(face["kps"], dtype=np.float32))
            scores.append(float(face.get("det_score", 0.9)))

    if not embeds:
        raise ValueError("Nenhuma face detectada na 'reference_image' fornecida.")

    return embeds, kps_list, scores


def combine_reference_embeddings(embeds, scores, mode="norm_average_weighted"):
    """
    Combines multiple 512-d ArcFace/GlintR100 embeddings while preserving the exact L2 norm
    magnitude (~22-26) expected by InstantID's Perceiver Resampler.
    """
    stack = torch.stack(embeds, dim=0)  # (N, 512)
    if stack.shape[0] == 1:
        return stack.unsqueeze(0)  # (1, 1, 512)

    if mode == "concat":
        return stack.unsqueeze(1)  # (N, 1, 512)

    norms = torch.norm(stack, dim=-1, keepdim=True).clamp(min=1e-6)
    unit_embeds = stack / norms

    if mode == "norm_average_weighted":
        w = torch.tensor(scores, dtype=stack.dtype, device=stack.device).unsqueeze(-1)
        w = w.clamp(min=0.1) ** 2
        w = w / w.sum()
        mean_dir = torch.sum(unit_embeds * w, dim=0, keepdim=True)
        mean_norm = torch.sum(norms * w, dim=0, keepdim=True)
        mean_dir = mean_dir / torch.norm(mean_dir, dim=-1, keepdim=True).clamp(min=1e-6)
        combined = mean_dir * mean_norm
        return combined.unsqueeze(0)

    elif mode == "norm_average":
        mean_dir = torch.mean(unit_embeds, dim=0, keepdim=True)
        mean_norm = torch.mean(norms, dim=0, keepdim=True)
        mean_dir = mean_dir / torch.norm(mean_dir, dim=-1, keepdim=True).clamp(min=1e-6)
        combined = mean_dir * mean_norm
        return combined.unsqueeze(0)

    else:
        return torch.mean(stack, dim=0, keepdim=True).unsqueeze(0)


def build_canonical_reference_kps(kps_list, scores):
    """
    Builds a canonical 5-point facial geometry from the reference face(s) by aligning
    all frontal reference faces into a normalized coordinate system via Similarity Transform.
    """
    aligned_list = []
    weights = []

    for kps, score in zip(kps_list, scores):
        kps = np.asarray(kps, dtype=np.float32)
        frontal = compute_face_frontalness(kps)
        w = max(0.05, frontal * float(score))

        left_eye, right_eye = kps[0], kps[1]
        eye_vec = right_eye - left_eye
        dist = float(np.linalg.norm(eye_vec))
        if dist < 1e-3:
            continue

        scale = 1.0 / dist
        angle = math.atan2(float(eye_vec[1]), float(eye_vec[0]))
        cos_a, sin_a = math.cos(-angle), math.sin(-angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32) * scale
        eye_mid = 0.5 * (left_eye + right_eye)
        kps_norm = (kps - eye_mid) @ R.T
        aligned_list.append(kps_norm)
        weights.append(w)

    if not aligned_list:
        return np.asarray(kps_list[0], dtype=np.float32)

    weights = np.array(weights, dtype=np.float32)
    weights /= np.sum(weights)
    canon_kps = np.sum(np.stack(aligned_list, axis=0) * weights[:, None, None], axis=0)
    return canon_kps


def morph_keypoints_to_identity(
    target_kps, ref_kps_list, ref_scores, morph_weight=0.65, mouth_keep=0.0, nose_keep=0.0
):
    """
    Projects the reference person's canonical 5-keypoint facial proportions onto the target
    face's exact position, scale, and 3D head pose.
    When mouth_keep or nose_keep > 0, preserves the base image's exact mouth/nose keypoint positions
    so open mouths, sticking-out tongues, cigarettes, or nose piercings are not geometrically crushed.
    """
    target_kps = np.asarray(target_kps, dtype=np.float32)
    if morph_weight <= 0.0 or not ref_kps_list:
        return target_kps.copy()

    canon_ref = build_canonical_reference_kps(ref_kps_list, ref_scores)

    M, _ = cv2.estimateAffinePartial2D(canon_ref, target_kps, method=cv2.LMEDS)
    if M is None:
        return target_kps.copy()

    ref_aligned = (canon_ref @ M[:, :2].T) + M[:, 2]

    eye_vec = target_kps[1] - target_kps[0]
    eye_dist = float(np.linalg.norm(eye_vec))
    if eye_dist < 1e-3:
        return target_kps.copy()

    ex = eye_vec / eye_dist
    ey = np.array([-ex[1], ex[0]], dtype=np.float32)

    frontalness = compute_face_frontalness(target_kps)

    delta = ref_aligned - target_kps
    delta_x = np.sum(delta * ex[None, :], axis=-1, keepdims=True) * ex[None, :]
    delta_y = np.sum(delta * ey[None, :], axis=-1, keepdims=True) * ey[None, :]

    w_x = float(morph_weight) * (frontalness ** 1.5)
    w_y = float(morph_weight) * max(0.45, frontalness)

    # Per-keypoint attenuation when preserving nose (idx 2) or mouth (idx 3, 4)
    kp_scales = np.ones((5, 1), dtype=np.float32)
    if nose_keep > 0.0:
        kp_scales[2, 0] = max(0.0, 1.0 - float(nose_keep) * 0.90)
    if mouth_keep > 0.0:
        kp_scales[3, 0] = max(0.0, 1.0 - float(mouth_keep) * 0.95)
        kp_scales[4, 0] = max(0.0, 1.0 - float(mouth_keep) * 0.95)

    morphed = target_kps + kp_scales * (w_x * delta_x + w_y * delta_y)
    return morphed.astype(np.float32)


def build_anatomical_face_mask(
    crop_size,
    face_dict,
    morphed_kps,
    mask_mode="landmarks_106_forehead",
    expand_px=16,
    blur_px=32,
    mouth_keep=0.0,
    nose_keep=0.0,
):
    """
    Builds a smooth float32 mask (H, W) in [0.0, 1.0] on the 1024x1024 crop using InsightFace's
    106 2D landmarks (plus morphed keypoints and anatomical forehead projection).
    Optionally carves smooth protection zones over the mouth/tongue/cigarette (`mouth_keep`)
    and/or nose/piercing (`nose_keep`).
    """
    h, w = crop_size, crop_size
    mask = np.zeros((h, w), dtype=np.uint8)

    kps = np.asarray(morphed_kps, dtype=np.float32)
    orig_kps = np.asarray(face_dict.get("kps", morphed_kps), dtype=np.float32)
    left_eye, right_eye, nose, mouth_l, mouth_r = kps[0], kps[1], kps[2], kps[3], kps[4]
    eye_mid = 0.5 * (left_eye + right_eye)
    mouth_mid = 0.5 * (mouth_l + mouth_r)

    eye_vec = right_eye - left_eye
    eye_dist = max(float(np.linalg.norm(eye_vec)), 1.0)
    ex = eye_vec / eye_dist
    ey = np.array([-ex[1], ex[0]], dtype=np.float32)
    if np.dot(mouth_mid - eye_mid, ey) < 0:
        ey = -ey

    face_pts = []
    lm106 = face_dict.get("landmark_2d_106", None)

    if lm106 is not None and mask_mode in ("landmarks_106_forehead", "landmarks_106_tight"):
        lm106 = np.asarray(lm106, dtype=np.float32)
        face_pts.append(lm106)
        face_pts.append(kps)

        proj_y = np.dot(lm106 - eye_mid[None, :], ey)
        proj_x = np.dot(lm106 - eye_mid[None, :], ex)
        chin_dist = float(np.max(proj_y))
        top_brow_dist = float(np.min(proj_y))
        half_width = float(np.max(np.abs(proj_x)))

        if mask_mode == "landmarks_106_forehead":
            forehead_lift = abs(top_brow_dist) + 0.34 * chin_dist
            arc_angles = np.linspace(-0.78, 0.78, 9)
            forehead_pts = []
            for ang in arc_angles:
                px = math.sin(ang) * (half_width * 0.90)
                py = -math.cos(ang) * forehead_lift
                pt = eye_mid + px * ex + py * ey
                forehead_pts.append(pt)
            face_pts.append(np.array(forehead_pts, dtype=np.float32))
        else:
            brow_lift = abs(top_brow_dist) + 0.08 * chin_dist
            for ang in np.linspace(-0.65, 0.65, 5):
                px = math.sin(ang) * (half_width * 0.80)
                py = -math.cos(ang) * brow_lift
                face_pts.append((eye_mid + px * ex + py * ey)[None, :])

        all_pts = np.concatenate(face_pts, axis=0)
        hull = cv2.convexHull(all_pts.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
    else:
        bbox = np.asarray(face_dict["bbox"], dtype=np.float32)
        cx, cy = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
        bw, bh = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        angle = math.degrees(math.atan2(float(ex[1]), float(ex[0])))
        axes = (int(bw * 0.48), int(bh * 0.54))
        cv2.ellipse(mask, (int(cx), int(cy)), axes, angle, 0, 360, 255, -1)

    if expand_px != 0:
        ksize = abs(int(expand_px)) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        if expand_px > 0:
            mask = cv2.dilate(mask, kernel, iterations=1)
        else:
            mask = cv2.erode(mask, kernel, iterations=1)

    border_margin = max(12, int(blur_px * 0.75))
    border_mask = np.zeros_like(mask)
    cv2.rectangle(
        border_mask,
        (border_margin, border_margin),
        (w - border_margin, h - border_margin),
        255,
        -1,
    )
    mask = cv2.bitwise_and(mask, border_mask)

    mask_f = mask.astype(np.float32) / 255.0
    if blur_px > 0:
        kblur = int(blur_px) * 2 + 1
        mask_f = cv2.GaussianBlur(mask_f, (kblur, kblur), float(blur_px) * 0.45)

    angle_deg = math.degrees(math.atan2(float(ex[1]), float(ex[0])))

    # Optional anatomical mouth / tongue / cigarette protection zone
    if mouth_keep > 0.0:
        mouth_prot = np.zeros((h, w), dtype=np.uint8)
        om_l, om_r = orig_kps[3], orig_kps[4]
        om_mid = 0.5 * (om_l + om_r)
        mouth_w = max(float(np.linalg.norm(om_r - om_l)), eye_dist * 0.72)

        if lm106 is not None and len(lm106) >= 72:
            lip_pts = lm106[52:72].copy()
            # Also include a slightly downward-shifted copy to cover sticking-out tongue or cigarette
            lip_down = lip_pts + (0.28 * mouth_w) * ey[None, :]
            hull_m = cv2.convexHull(np.concatenate([lip_pts, lip_down], axis=0).astype(np.int32))
            cv2.fillConvexPoly(mouth_prot, hull_m, 255)

        # Reinforce with an elliptical zone covering lips + lower extension (tongue/cigarette)
        ell_center = om_mid + (0.12 * mouth_w) * ey
        axes_m = (int(round(mouth_w * 0.68)), int(round(mouth_w * 0.48)))
        cv2.ellipse(
            mouth_prot,
            (int(round(ell_center[0])), int(round(ell_center[1]))),
            axes_m,
            angle_deg,
            0,
            360,
            255,
            -1,
        )
        dil_m = max(5, int(round(eye_dist * 0.08))) * 2 + 1
        mouth_prot = cv2.dilate(
            mouth_prot, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_m, dil_m)), iterations=1
        )
        blur_m = max(15, int(round(eye_dist * 0.22))) * 2 + 1
        mouth_prot_f = cv2.GaussianBlur(
            mouth_prot.astype(np.float32) / 255.0, (blur_m, blur_m), float(blur_m) * 0.25
        )
        mask_f = mask_f * (1.0 - np.clip(float(mouth_keep), 0.0, 1.0) * mouth_prot_f)

    # Optional anatomical nose / piercing protection zone
    if nose_keep > 0.0:
        nose_prot = np.zeros((h, w), dtype=np.uint8)
        onose = orig_kps[2]
        if lm106 is not None and len(lm106) >= 87:
            nose_pts = lm106[72:87].copy()
            # Expand slightly downward to cover septum rings
            nose_down = nose_pts + (0.10 * eye_dist) * ey[None, :]
            hull_n = cv2.convexHull(np.concatenate([nose_pts, nose_down], axis=0).astype(np.int32))
            cv2.fillConvexPoly(nose_prot, hull_n, 255)

        axes_n = (int(round(eye_dist * 0.34)), int(round(eye_dist * 0.30)))
        cv2.ellipse(
            nose_prot,
            (int(round(onose[0])), int(round(onose[1]))),
            axes_n,
            angle_deg,
            0,
            360,
            255,
            -1,
        )
        dil_n = max(3, int(round(eye_dist * 0.05))) * 2 + 1
        nose_prot = cv2.dilate(
            nose_prot, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_n, dil_n)), iterations=1
        )
        blur_n = max(13, int(round(eye_dist * 0.18))) * 2 + 1
        nose_prot_f = cv2.GaussianBlur(
            nose_prot.astype(np.float32) / 255.0, (blur_n, blur_n), float(blur_n) * 0.25
        )
        mask_f = mask_f * (1.0 - np.clip(float(nose_keep), 0.0, 1.0) * nose_prot_f)

    return np.clip(mask_f, 0.0, 1.0)


def match_skin_color_lab(src_rgb_u8, target_rgb_u8, mask_f, mode="LAB_skin_match"):
    """
    Matches the skin tone and luminance of src_rgb_u8 to target_rgb_u8 in perceptual CIE LAB space.
    """
    if mode == "none":
        return src_rgb_u8

    core = mask_f > (0.45 if mode == "LAB_skin_match" else 0.15)
    if np.sum(core) < 100:
        return src_rgb_u8

    src_lab = cv2.cvtColor(src_rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)

    out_lab = src_lab.copy()
    chan_weights = [0.85, 0.95, 0.95]

    for c in range(3):
        s_vals = src_lab[:, :, c][core]
        t_vals = tgt_lab[:, :, c][core]

        s_mean, s_std = float(np.mean(s_vals)), float(np.std(s_vals)) + 1e-5
        t_mean, t_std = float(np.mean(t_vals)), float(np.std(t_vals)) + 1e-5

        std_ratio = np.clip(t_std / s_std, 0.65, 1.45)
        transferred = (src_lab[:, :, c] - s_mean) * std_ratio + t_mean

        w = chan_weights[c]
        out_lab[:, :, c] = (1.0 - w) * src_lab[:, :, c] + w * transferred

    out_lab = np.clip(out_lab, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)


def transfer_high_frequency_skin_detail(refined_rgb_u8, moss_orig_rgb_u8, mask_f, morphed_kps, strength=0.45):
    """
    Frequency-separated skin pore & sensor grain transfer tailored for MOSS-Image SFT (Flux2 32-ch VAE).
    """
    if strength <= 0.0:
        return refined_rgb_u8

    h, w, _ = refined_rgb_u8.shape
    ref_lab = cv2.cvtColor(refined_rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    moss_lab = cv2.cvtColor(moss_orig_rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)

    L_ref = ref_lab[:, :, 0]
    L_moss = moss_lab[:, :, 0]

    L_ref_low = cv2.GaussianBlur(L_ref, (9, 9), 1.8)
    L_moss_low = cv2.GaussianBlur(L_moss, (9, 9), 1.8)

    hf_ref = L_ref - L_ref_low
    hf_moss = np.clip(L_moss - L_moss_low, -5.5, 5.5)

    gx = cv2.Sobel(L_moss_low, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L_moss_low, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    edge_gate = np.exp(-((grad_mag / 18.0) ** 2))

    kps = np.asarray(morphed_kps, dtype=np.float32)
    eye_dist = max(float(np.linalg.norm(kps[1] - kps[0])), 32.0)
    r_eye = int(round(eye_dist * 0.26))
    r_nose = int(round(eye_dist * 0.20))
    r_mouth = int(round(eye_dist * 0.22))

    excl_draw = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(excl_draw, (int(kps[0][0]), int(kps[0][1])), r_eye, 255, -1)
    cv2.circle(excl_draw, (int(kps[1][0]), int(kps[1][1])), r_eye, 255, -1)
    cv2.circle(excl_draw, (int(kps[2][0]), int(kps[2][1])), r_nose, 255, -1)
    cv2.line(
        excl_draw,
        (int(kps[3][0]), int(kps[3][1])),
        (int(kps[4][0]), int(kps[4][1])),
        255,
        thickness=r_mouth * 2,
    )
    excl_blur = cv2.GaussianBlur(excl_draw.astype(np.float32) / 255.0, (31, 31), 10.0)
    skin_gate = np.clip(1.0 - excl_blur, 0.0, 1.0) * edge_gate * np.clip(mask_f, 0.0, 1.0)

    w_hf = float(strength) * skin_gate
    ref_lab[:, :, 0] = np.clip(L_ref + w_hf * (hf_moss - hf_ref), 0.0, 255.0)

    return cv2.cvtColor(ref_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def apply_differential_diffusion(model, strength=1.0):
    work_model = model.clone()

    def forward_denoise_mask(sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict):
        m = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = m.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]

        ts_from = m.inner_model.model_sampling.timestep(sigma_from)
        ts_to = m.inner_model.model_sampling.timestep(sigma_to)
        current_ts = m.inner_model.model_sampling.timestep(sigma[0])

        denom = ts_from - ts_to
        if abs(float(denom)) < 1e-5:
            threshold = 0.0
        else:
            threshold = (current_ts - ts_to) / denom

        binary_mask = (denoise_mask >= threshold).to(denoise_mask.dtype)
        if strength and strength < 1.0:
            return strength * binary_mask + (1.0 - strength) * denoise_mask
        return binary_mask

    work_model.set_model_denoise_mask_function(forward_denoise_mask)
    return work_model


def _prepare_ref_slot_bundle(insightface, instantid, device, dtype, use_folder, folder_path, ref_image):
    """
    Loads reference embeddings & keypoints for a reference slot and computes its InstantID Resampler tokens.
    """
    ref_embeds, ref_kps_list, ref_scores = extract_reference_data(
        insightface, use_folder, folder_path, ref_image
    )
    clip_embed = combine_reference_embeddings(ref_embeds, ref_scores, mode="norm_average_weighted")
    clip_embed_zeroed = torch.zeros_like(clip_embed)

    instantid.to(device, dtype=dtype)
    image_prompt_embeds, uncond_image_prompt_embeds = instantid.get_image_embeds(
        clip_embed.to(device, dtype=dtype),
        clip_embed_zeroed.to(device, dtype=dtype),
    )
    return {
        "cond": image_prompt_embeds.to(device, dtype=dtype),
        "uncond": uncond_image_prompt_embeds.to(device, dtype=dtype),
        "kps_list": ref_kps_list,
        "scores": ref_scores,
    }


TARGET_GENDER_OPTIONS = [
    "all (todos)",
    "female_only (apenas mulher)",
    "male_only (apenas homem)",
    "couple_both (casal: ref1=mulher, ref2=homem)",
]

TARGET_POSITION_OPTIONS = [
    "largest (maior / principal)",
    "all (todas as faces)",
    "center (a do meio)",
    "left_to_right (esquerda -> direita)",
    "right_to_left (direita -> esquerda)",
    "top_to_bottom (cima -> baixo)",
    "bottom_to_top (baixo -> cima)",
]


# ==============================================================================
# Node 1: InstantIDKrea2FaceRefiner
# ==============================================================================
class InstantIDKrea2FaceRefiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "instantid": ("INSTANTID",),
                "insightface": ("FACEANALYSIS",),
                "control_net": ("CONTROL_NET",),
                "sdxl_model": ("MODEL",),
                "sdxl_clip": ("CLIP",),
                "sdxl_vae": ("VAE",),
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "(hat:2.0), bad anatomy, closed eyes, deformed face, blurry, out of focus, bad skin, plastic skin, asymmetrical eyes",
                    },
                ),
                "use_reference_folder": ("BOOLEAN", {"default": True}),
                "reference_folder": ("STRING", {"default": "D:\\trabalhos de IA\\archive\\faces\\Lais"}),
                "target_gender": (TARGET_GENDER_OPTIONS, {"default": "all (todos)"}),
                "target_position": (TARGET_POSITION_OPTIONS, {"default": "largest (maior / principal)"}),
                "target_index": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
                "use_reference_folder_2": ("BOOLEAN", {"default": True}),
                "reference_folder_2": ("STRING", {"default": ""}),
                "preset_profile": (
                    get_preset_options_list(),
                    {"default": "manual (usar controles abaixo)"},
                ),
                "ip_weight": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 3.0, "step": 0.01}),
                "cn_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 3.0, "step": 0.01}),
                "kps_identity_morph": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "crop_scale": ("FLOAT", {"default": 2.2, "min": 1.3, "max": 4.0, "step": 0.05}),
                "denoise": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "lcm"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "exponential"}),
                "mask_mode": (
                    ["landmarks_106_forehead", "landmarks_106_tight", "face_oval"],
                    {"default": "landmarks_106_forehead"},
                ),
                "mask_expand": ("INT", {"default": 16, "min": -64, "max": 128, "step": 1}),
                "mask_blur": ("INT", {"default": 32, "min": 1, "max": 150, "step": 1}),
                "mouth_mask_keep": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "nose_mask_keep": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "color_match": (
                    ["LAB_skin_match", "LAB_full_match", "none"],
                    {"default": "LAB_skin_match"},
                ),
                "free_vram_before": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "crop_mask_preview", "kps_preview", "face_mask")
    FUNCTION = "refine_face"
    CATEGORY = "InstantID"

    def refine_face(
        self,
        base_image,
        instantid,
        insightface,
        control_net,
        sdxl_model,
        sdxl_clip,
        sdxl_vae,
        positive_prompt,
        negative_prompt,
        use_reference_folder,
        reference_folder,
        target_gender="all (todos)",
        target_position="largest (maior / principal)",
        target_index=1,
        use_reference_folder_2=True,
        reference_folder_2="",
        preset_profile="manual (usar controles abaixo)",
        ip_weight=0.95,
        cn_strength=0.85,
        kps_identity_morph=0.65,
        crop_scale=2.2,
        denoise=0.55,
        seed=0,
        steps=12,
        cfg=1.0,
        sampler_name="lcm",
        scheduler="exponential",
        mask_mode="landmarks_106_forehead",
        mask_expand=16,
        mask_blur=32,
        mouth_mask_keep=0.0,
        nose_mask_keep=0.0,
        color_match="LAB_skin_match",
        free_vram_before=True,
        reference_image=None,
        reference_image_2=None,
        **kwargs,
    ):
        return self._execute_pipeline(
            base_image=base_image,
            instantid=instantid,
            insightface=insightface,
            control_net=control_net,
            sdxl_model=sdxl_model,
            sdxl_clip=sdxl_clip,
            sdxl_vae=sdxl_vae,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            use_reference_folder=use_reference_folder,
            reference_folder=reference_folder,
            target_gender=target_gender,
            target_position=target_position,
            target_index=target_index,
            use_reference_folder_2=use_reference_folder_2,
            reference_folder_2=reference_folder_2,
            preset_profile=preset_profile,
            ip_weight=ip_weight,
            cn_strength=cn_strength,
            kps_identity_morph=kps_identity_morph,
            crop_scale=crop_scale,
            denoise=denoise,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            mask_mode=mask_mode,
            mask_expand=mask_expand,
            mask_blur=mask_blur,
            mouth_mask_keep=mouth_mask_keep,
            nose_mask_keep=nose_mask_keep,
            color_match=color_match,
            hf_pore_transfer=0.0,
            moss_polish_denoise=0.0,
            moss_polish_steps=6,
            moss_polish_cfg=1.0,
            free_vram_before=free_vram_before,
            reference_image=reference_image,
            reference_image_2=reference_image_2,
            **kwargs,
        )

    def _execute_pipeline(
        self,
        base_image,
        instantid,
        insightface,
        control_net,
        sdxl_model,
        sdxl_clip,
        sdxl_vae,
        positive_prompt,
        negative_prompt,
        use_reference_folder,
        reference_folder,
        target_gender="all (todos)",
        target_position="largest (maior / principal)",
        target_index=1,
        use_reference_folder_2=True,
        reference_folder_2="",
        preset_profile="manual (usar controles abaixo)",
        ip_weight=0.95,
        cn_strength=0.85,
        kps_identity_morph=0.65,
        crop_scale=2.2,
        denoise=0.55,
        seed=0,
        steps=12,
        cfg=1.0,
        sampler_name="lcm",
        scheduler="exponential",
        mask_mode="landmarks_106_forehead",
        mask_expand=16,
        mask_blur=32,
        mouth_mask_keep=0.0,
        nose_mask_keep=0.0,
        color_match="LAB_skin_match",
        hf_pore_transfer=0.0,
        moss_polish_denoise=0.0,
        moss_polish_steps=6,
        moss_polish_cfg=1.0,
        free_vram_before=True,
        reference_image=None,
        reference_image_2=None,
        moss_model=None,
        moss_positive=None,
        moss_negative=None,
        moss_vae=None,
        **kwargs,
    ):
        # Apply preset overrides if a non-manual preset profile is selected
        if preset_profile and not preset_profile.startswith("manual"):
            all_presets = get_all_presets()
            p_vals = all_presets.get(preset_profile, {})
            if p_vals:
                ip_weight = float(p_vals.get("ip_weight", ip_weight))
                cn_strength = float(p_vals.get("cn_strength", cn_strength))
                kps_identity_morph = float(p_vals.get("kps_identity_morph", kps_identity_morph))
                crop_scale = float(p_vals.get("crop_scale", crop_scale))
                denoise = float(p_vals.get("denoise", denoise))
                steps = int(p_vals.get("steps", steps))
                cfg = float(p_vals.get("cfg", cfg))
                mask_expand = int(p_vals.get("mask_expand", mask_expand))
                mask_blur = int(p_vals.get("mask_blur", mask_blur))
                mouth_mask_keep = float(p_vals.get("mouth_mask_keep", mouth_mask_keep))
                nose_mask_keep = float(p_vals.get("nose_mask_keep", nose_mask_keep))
                if hf_pore_transfer > 0.0 and "hf_pore_transfer" in p_vals:
                    hf_pore_transfer = float(p_vals["hf_pore_transfer"])
                print(
                    f"\033[36m[InstantID Hybrid] Preset ativo: '{preset_profile}' -> ip={ip_weight}, cn={cn_strength}, morph={kps_identity_morph}, denoise={denoise}, mouth_keep={mouth_mask_keep}, nose_keep={nose_mask_keep}\033[0m"
                )

        # Defensive sanitization against shifted or invalid widget inputs
        try:
            seed = int(seed)
        except Exception:
            seed = int(torch.randint(0, 0xFFFFFFFFFFFFFFFF, (1,)).item())

        if sampler_name not in comfy.samplers.KSampler.SAMPLERS:
            sampler_name = "lcm"
        if scheduler not in comfy.samplers.KSampler.SCHEDULERS:
            scheduler = "exponential"
        if mask_mode not in ["landmarks_106_forehead", "landmarks_106_tight", "face_oval"]:
            mask_mode = "landmarks_106_forehead"
        if color_match not in ["LAB_skin_match", "LAB_full_match", "none"]:
            color_match = "LAB_skin_match"
        try:
            steps = max(1, int(steps))
        except Exception:
            steps = 15
        try:
            cfg = float(cfg)
        except Exception:
            cfg = 1.0
        try:
            denoise = float(denoise)
        except Exception:
            denoise = 0.55
        try:
            ip_weight = float(ip_weight)
        except Exception:
            ip_weight = 0.95
        try:
            cn_strength = float(cn_strength)
        except Exception:
            cn_strength = 0.85
        try:
            kps_identity_morph = float(kps_identity_morph)
        except Exception:
            kps_identity_morph = 0.65
        try:
            crop_scale = float(crop_scale)
        except Exception:
            crop_scale = 2.0
        try:
            mask_expand = int(mask_expand)
        except Exception:
            mask_expand = 16
        try:
            mask_blur = int(mask_blur)
        except Exception:
            mask_blur = 32
        try:
            mouth_mask_keep = float(mouth_mask_keep)
        except Exception:
            mouth_mask_keep = 0.0
        try:
            nose_mask_keep = float(nose_mask_keep)
        except Exception:
            nose_mask_keep = 0.0
        try:
            hf_pore_transfer = float(hf_pore_transfer)
        except Exception:
            hf_pore_transfer = 0.45
        try:
            moss_polish_denoise = float(moss_polish_denoise)
        except Exception:
            moss_polish_denoise = 0.0
        try:
            moss_polish_steps = int(moss_polish_steps)
        except Exception:
            moss_polish_steps = 6
        try:
            moss_polish_cfg = float(moss_polish_cfg)
        except Exception:
            moss_polish_cfg = 1.0
        if free_vram_before:
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()

        dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        device = comfy.model_management.get_torch_device()

        # Load primary reference identity (Slot 1)
        ref_bundle_1 = _prepare_ref_slot_bundle(
            insightface,
            instantid,
            device,
            dtype,
            use_reference_folder,
            reference_folder,
            reference_image,
        )

        # Load secondary reference identity (Slot 2) if couple_both is selected and Ref 2 is provided
        ref_bundle_2 = ref_bundle_1
        if target_gender.startswith("couple_both"):
            f2_clean = (reference_folder_2 or "").strip().strip('"').strip("'")
            has_ref2 = (use_reference_folder_2 and bool(f2_clean and os.path.isdir(f2_clean))) or (
                reference_image_2 is not None
            )
            if has_ref2:
                try:
                    ref_bundle_2 = _prepare_ref_slot_bundle(
                        insightface,
                        instantid,
                        device,
                        dtype,
                        use_reference_folder_2,
                        reference_folder_2,
                        reference_image_2,
                    )
                except Exception as e:
                    print(
                        f"\033[33m[InstantID Hybrid] Aviso ao carregar Referência 2 ({e}); usando Referência 1.\033[0m"
                    )
                    ref_bundle_2 = ref_bundle_1

        ref_bundles = {1: ref_bundle_1, 2: ref_bundle_2}

        tokens_pos = sdxl_clip.tokenize(positive_prompt or "")
        pos_cond = sdxl_clip.encode_from_tokens_scheduled(tokens_pos)
        tokens_neg = sdxl_clip.tokenize(negative_prompt or "")
        neg_cond = sdxl_clip.encode_from_tokens_scheduled(tokens_neg)

        batch_size, img_h, img_w, _ = base_image.shape
        out_images = []
        out_previews = []
        out_kps = []
        out_masks = []

        target_res = 1024

        for b_idx in range(batch_size):
            single_tensor = base_image[b_idx : b_idx + 1]
            final_rgb_u8 = (single_tensor[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            img_bgr_u8 = final_rgb_u8[:, :, ::-1].copy()

            all_faces = detect_all_faces(insightface, img_bgr_u8)
            selected_targets = select_target_faces(
                all_faces,
                img_w,
                img_h,
                target_gender=target_gender,
                target_position=target_position,
                target_index=target_index,
            )

            if not selected_targets:
                print(
                    "\033[33m[InstantID Hybrid] Aviso: Nenhuma face correspondente ao filtro detectada na imagem base. Retornando imagem original.\033[0m"
                )
                out_images.append(single_tensor[0])
                out_previews.append(torch.zeros((target_res, target_res, 3), dtype=torch.float32))
                out_kps.append(torch.zeros((target_res, target_res, 3), dtype=torch.float32))
                out_masks.append(torch.zeros((img_h, img_w), dtype=torch.float32))
                continue

            full_mask_f = np.zeros((img_h, img_w), dtype=np.float32)

            for f_idx, (face_base, ref_slot) in enumerate(selected_targets):
                bundle = ref_bundles.get(ref_slot, ref_bundle_1)
                image_prompt_embeds = bundle["cond"]
                uncond_image_prompt_embeds = bundle["uncond"]
                ref_kps_list = bundle["kps_list"]
                ref_scores = bundle["scores"]

                x1, y1, x2, y2 = [float(v) for v in face_base["bbox"]]
                fw, fh = x2 - x1, y2 - y1
                fcx, fcy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

                box_size = int(round(max(fw, fh) * float(crop_scale)))
                box_size = max(256, min(box_size, min(img_w, img_h)))

                cx1 = int(round(fcx - box_size * 0.5))
                cy1 = int(round(fcy - box_size * 0.5))
                cx1 = max(0, min(cx1, img_w - box_size))
                cy1 = max(0, min(cy1, img_h - box_size))
                cx2 = cx1 + box_size
                cy2 = cy1 + box_size

                crop_rgb_u8 = final_rgb_u8[cy1:cy2, cx1:cx2].copy()
                crop_1024_rgb = cv2.resize(
                    crop_rgb_u8, (target_res, target_res), interpolation=cv2.INTER_LANCZOS4
                )
                crop_1024_bgr = crop_1024_rgb[:, :, ::-1].copy()

                scale_to_1024 = float(target_res) / float(box_size)
                # Detect inside crop and pick the face closest to the target face's center in crop coordinates
                target_crop_cx = (fcx - cx1) * scale_to_1024
                target_crop_cy = (fcy - cy1) * scale_to_1024
                crop_faces = detect_all_faces(insightface, crop_1024_bgr, min_score=0.25)

                if crop_faces:
                    face_crop = min(
                        crop_faces,
                        key=lambda cf: (_face_cx(cf) - target_crop_cx) ** 2
                        + (_face_cy(cf) - target_crop_cy) ** 2,
                    )
                else:
                    face_crop = {
                        "bbox": [
                            (x1 - cx1) * scale_to_1024,
                            (y1 - cy1) * scale_to_1024,
                            (x2 - cx1) * scale_to_1024,
                            (y2 - cy1) * scale_to_1024,
                        ],
                        "kps": (
                            np.asarray(face_base["kps"], dtype=np.float32)
                            - np.array([[cx1, cy1]], dtype=np.float32)
                        )
                        * scale_to_1024,
                    }
                    if "landmark_2d_106" in face_base and face_base["landmark_2d_106"] is not None:
                        face_crop["landmark_2d_106"] = (
                            np.asarray(face_base["landmark_2d_106"], dtype=np.float32)
                            - np.array([[cx1, cy1]], dtype=np.float32)
                        ) * scale_to_1024

                morphed_kps = morph_keypoints_to_identity(
                    face_crop["kps"],
                    ref_kps_list,
                    ref_scores,
                    morph_weight=kps_identity_morph,
                    mouth_keep=mouth_mask_keep,
                    nose_keep=nose_mask_keep,
                )

                kps_pil = draw_kps(crop_1024_bgr, morphed_kps)
                kps_np = np.array(kps_pil, dtype=np.float32) / 255.0
                face_kps_tensor = torch.from_numpy(kps_np).unsqueeze(0)

                mask_1024_f = build_anatomical_face_mask(
                    target_res,
                    face_crop,
                    morphed_kps,
                    mask_mode=mask_mode,
                    expand_px=mask_expand,
                    blur_px=mask_blur,
                    mouth_keep=mouth_mask_keep,
                    nose_keep=nose_mask_keep,
                )
                mask_1024_tensor = torch.from_numpy(mask_1024_f).unsqueeze(0)

                work_model = sdxl_model.clone()
                sigma_start = work_model.get_model_object("model_sampling").percent_to_sigma(0.0)
                sigma_end = work_model.get_model_object("model_sampling").percent_to_sigma(1.0)

                patch_kwargs = {
                    "ipadapter": instantid,
                    "weight": float(ip_weight),
                    "cond": image_prompt_embeds,
                    "uncond": uncond_image_prompt_embeds,
                    "mask": None,
                    "sigma_start": sigma_start,
                    "sigma_end": sigma_end,
                }

                number = 0
                for block_id in [4, 5, 7, 8]:
                    block_indices = range(2) if block_id in [4, 5] else range(10)
                    for index in block_indices:
                        patch_kwargs["module_key"] = str(number * 2 + 1)
                        _set_model_patch_replace(work_model, patch_kwargs, ("input", block_id, index))
                        number += 1
                for block_id in range(6):
                    block_indices = range(2) if block_id in [3, 4, 5] else range(10)
                    for index in block_indices:
                        patch_kwargs["module_key"] = str(number * 2 + 1)
                        _set_model_patch_replace(work_model, patch_kwargs, ("output", block_id, index))
                        number += 1
                for index in range(10):
                    patch_kwargs["module_key"] = str(number * 2 + 1)
                    _set_model_patch_replace(work_model, patch_kwargs, ("middle", 0, index))
                    number += 1

                work_model = apply_differential_diffusion(work_model, strength=1.0)

                cnets = {}
                cond_uncond = []
                is_cond = True
                for conditioning in [pos_cond, neg_cond]:
                    c = []
                    for t in conditioning:
                        d = t[1].copy()
                        prev_cnet = d.get("control", None)
                        if prev_cnet in cnets:
                            c_net = cnets[prev_cnet]
                        else:
                            c_net = control_net.copy().set_cond_hint(
                                face_kps_tensor.movedim(-1, 1),
                                float(cn_strength),
                                (0.0, 1.0),
                            )
                            c_net.set_previous_controlnet(prev_cnet)
                            cnets[prev_cnet] = c_net

                        d["control"] = c_net
                        d["control_apply_to_uncond"] = False
                        d["cross_attn_controlnet"] = (
                            image_prompt_embeds.to(comfy.model_management.intermediate_device())
                            if is_cond
                            else uncond_image_prompt_embeds.to(
                                comfy.model_management.intermediate_device()
                            )
                        )
                        c.append([t[0], d])
                    cond_uncond.append(c)
                    is_cond = False

                crop_1024_tensor = torch.from_numpy(
                    crop_1024_rgb.astype(np.float32) / 255.0
                ).unsqueeze(0)
                orig_latent = sdxl_vae.encode(crop_1024_tensor[:, :, :, :3])
                latent_dict = {
                    "samples": orig_latent,
                    "noise_mask": mask_1024_tensor,
                }

                sampled_latent = nodes.common_ksampler(
                    work_model,
                    seed + b_idx * 100 + f_idx,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    cond_uncond[0],
                    cond_uncond[1],
                    latent_dict,
                    denoise=float(denoise),
                )[0]

                decoded_crop = sdxl_vae.decode(sampled_latent["samples"])
                refined_1024_rgb = (
                    (decoded_crop[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                )

                matched_1024_rgb = match_skin_color_lab(
                    refined_1024_rgb, crop_1024_rgb, mask_1024_f, mode=color_match
                )
                if hf_pore_transfer > 0.0:
                    matched_1024_rgb = transfer_high_frequency_skin_detail(
                        matched_1024_rgb,
                        crop_1024_rgb,
                        mask_1024_f,
                        morphed_kps,
                        strength=float(hf_pore_transfer),
                    )

                alpha_1024 = mask_1024_f[:, :, None]
                blended_1024_rgb = (
                    crop_1024_rgb.astype(np.float32) * (1.0 - alpha_1024)
                    + matched_1024_rgb.astype(np.float32) * alpha_1024
                ).clip(0, 255).astype(np.uint8)

                # Optional MOSS-Image SFT + Flux2-VAE 32-ch Latent Micro-Polish
                if (
                    moss_polish_denoise > 0.0
                    and moss_model is not None
                    and moss_vae is not None
                    and moss_positive is not None
                    and moss_negative is not None
                ):
                    if free_vram_before:
                        comfy.model_management.unload_all_models()
                        comfy.model_management.soft_empty_cache()

                    moss_crop_tensor = torch.from_numpy(
                        blended_1024_rgb.astype(np.float32) / 255.0
                    ).unsqueeze(0)
                    moss_orig_latent = moss_vae.encode(moss_crop_tensor[:, :, :, :3])
                    moss_latent_dict = {
                        "samples": moss_orig_latent,
                        "noise_mask": mask_1024_tensor,
                    }
                    moss_work_model = apply_differential_diffusion(moss_model, strength=1.0)
                    moss_sampled = nodes.common_ksampler(
                        moss_work_model,
                        seed + b_idx * 100 + f_idx + 777,
                        int(moss_polish_steps),
                        float(moss_polish_cfg),
                        "euler",
                        "simple",
                        moss_positive,
                        moss_negative,
                        moss_latent_dict,
                        denoise=float(moss_polish_denoise),
                    )[0]
                    moss_decoded = moss_vae.decode(moss_sampled["samples"])
                    moss_polished_rgb = (
                        (moss_decoded[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    )
                    moss_polished_rgb = match_skin_color_lab(
                        moss_polished_rgb, crop_1024_rgb, mask_1024_f, mode=color_match
                    )
                    blended_1024_rgb = (
                        crop_1024_rgb.astype(np.float32) * (1.0 - alpha_1024)
                        + moss_polished_rgb.astype(np.float32) * alpha_1024
                    ).clip(0, 255).astype(np.uint8)

                blended_box_rgb = cv2.resize(
                    blended_1024_rgb, (box_size, box_size), interpolation=cv2.INTER_LANCZOS4
                )
                mask_box_f = cv2.resize(
                    mask_1024_f, (box_size, box_size), interpolation=cv2.INTER_LINEAR
                )[:, :, None]

                roi_orig = final_rgb_u8[cy1:cy2, cx1:cx2].astype(np.float32)
                roi_comp = (
                    roi_orig * (1.0 - mask_box_f) + blended_box_rgb.astype(np.float32) * mask_box_f
                )
                final_rgb_u8[cy1:cy2, cx1:cx2] = roi_comp.clip(0, 255).astype(np.uint8)

                full_mask_f[cy1:cy2, cx1:cx2] = np.maximum(
                    full_mask_f[cy1:cy2, cx1:cx2], mask_box_f[:, :, 0]
                )

                preview_rgb = blended_1024_rgb.copy()
                contour_bin = (mask_1024_f > 0.35).astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    contour_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                border_color = (0, 255, 128) if ref_slot == 1 else (255, 180, 0)
                cv2.drawContours(preview_rgb, contours, -1, border_color, 2)

                out_previews.append(torch.from_numpy(preview_rgb.astype(np.float32) / 255.0))
                out_kps.append(face_kps_tensor[0].cpu())

            out_images.append(torch.from_numpy(final_rgb_u8.astype(np.float32) / 255.0))
            out_masks.append(torch.from_numpy(full_mask_f))

        return (
            torch.stack(out_images, dim=0),
            torch.stack(out_previews, dim=0),
            torch.stack(out_kps, dim=0),
            torch.stack(out_masks, dim=0),
        )


# ==============================================================================
# Node 2: InstantIDMossFaceRefiner
# ==============================================================================
class InstantIDMossFaceRefiner(InstantIDKrea2FaceRefiner):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "instantid": ("INSTANTID",),
                "insightface": ("FACEANALYSIS",),
                "control_net": ("CONTROL_NET",),
                "sdxl_model": ("MODEL",),
                "sdxl_clip": ("CLIP",),
                "sdxl_vae": ("VAE",),
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "(hat:2.0), bad anatomy, closed eyes, deformed face, blurry, out of focus, bad skin, plastic skin, asymmetrical eyes",
                    },
                ),
                "use_reference_folder": ("BOOLEAN", {"default": True}),
                "reference_folder": ("STRING", {"default": "D:\\trabalhos de IA\\archive\\faces\\Lais"}),
                "target_gender": (TARGET_GENDER_OPTIONS, {"default": "all (todos)"}),
                "target_position": (TARGET_POSITION_OPTIONS, {"default": "largest (maior / principal)"}),
                "target_index": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
                "use_reference_folder_2": ("BOOLEAN", {"default": True}),
                "reference_folder_2": ("STRING", {"default": ""}),
                "preset_profile": (
                    get_preset_options_list(),
                    {"default": "manual (usar controles abaixo)"},
                ),
                "ip_weight": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 3.0, "step": 0.01}),
                "cn_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 3.0, "step": 0.01}),
                "kps_identity_morph": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "crop_scale": ("FLOAT", {"default": 2.2, "min": 1.3, "max": 4.0, "step": 0.05}),
                "denoise": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "lcm"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "exponential"}),
                "mask_mode": (
                    ["landmarks_106_forehead", "landmarks_106_tight", "face_oval"],
                    {"default": "landmarks_106_forehead"},
                ),
                "mask_expand": ("INT", {"default": 16, "min": -64, "max": 128, "step": 1}),
                "mask_blur": ("INT", {"default": 32, "min": 1, "max": 150, "step": 1}),
                "mouth_mask_keep": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "nose_mask_keep": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "color_match": (
                    ["LAB_skin_match", "LAB_full_match", "none"],
                    {"default": "LAB_skin_match"},
                ),
                "hf_pore_transfer": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.05}),
                "moss_polish_denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.45, "step": 0.01}),
                "moss_polish_steps": ("INT", {"default": 6, "min": 1, "max": 30}),
                "moss_polish_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 15.0, "step": 0.1}),
                "free_vram_before": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "moss_model": ("MODEL",),
                "moss_positive": ("CONDITIONING",),
                "moss_negative": ("CONDITIONING",),
                "moss_vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "crop_mask_preview", "kps_preview", "face_mask")
    FUNCTION = "refine_face_moss"
    CATEGORY = "InstantID"

    def refine_face_moss(
        self,
        base_image,
        instantid,
        insightface,
        control_net,
        sdxl_model,
        sdxl_clip,
        sdxl_vae,
        positive_prompt,
        negative_prompt,
        use_reference_folder,
        reference_folder,
        target_gender="all (todos)",
        target_position="largest (maior / principal)",
        target_index=1,
        use_reference_folder_2=True,
        reference_folder_2="",
        preset_profile="manual (usar controles abaixo)",
        ip_weight=0.95,
        cn_strength=0.85,
        kps_identity_morph=0.65,
        crop_scale=2.2,
        denoise=0.55,
        seed=0,
        steps=12,
        cfg=1.0,
        sampler_name="lcm",
        scheduler="exponential",
        mask_mode="landmarks_106_forehead",
        mask_expand=16,
        mask_blur=32,
        mouth_mask_keep=0.0,
        nose_mask_keep=0.0,
        color_match="LAB_skin_match",
        hf_pore_transfer=0.45,
        moss_polish_denoise=0.0,
        moss_polish_steps=6,
        moss_polish_cfg=1.0,
        free_vram_before=True,
        reference_image=None,
        reference_image_2=None,
        moss_model=None,
        moss_positive=None,
        moss_negative=None,
        moss_vae=None,
        **kwargs,
    ):
        return self._execute_pipeline(
            base_image=base_image,
            instantid=instantid,
            insightface=insightface,
            control_net=control_net,
            sdxl_model=sdxl_model,
            sdxl_clip=sdxl_clip,
            sdxl_vae=sdxl_vae,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            use_reference_folder=use_reference_folder,
            reference_folder=reference_folder,
            target_gender=target_gender,
            target_position=target_position,
            target_index=target_index,
            use_reference_folder_2=use_reference_folder_2,
            reference_folder_2=reference_folder_2,
            preset_profile=preset_profile,
            ip_weight=ip_weight,
            cn_strength=cn_strength,
            kps_identity_morph=kps_identity_morph,
            crop_scale=crop_scale,
            denoise=denoise,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            mask_mode=mask_mode,
            mask_expand=mask_expand,
            mask_blur=mask_blur,
            mouth_mask_keep=mouth_mask_keep,
            nose_mask_keep=nose_mask_keep,
            color_match=color_match,
            hf_pore_transfer=hf_pore_transfer,
            moss_polish_denoise=moss_polish_denoise,
            moss_polish_steps=moss_polish_steps,
            moss_polish_cfg=moss_polish_cfg,
            free_vram_before=free_vram_before,
            reference_image=reference_image,
            reference_image_2=reference_image_2,
            moss_model=moss_model,
            moss_positive=moss_positive,
            moss_negative=moss_negative,
            moss_vae=moss_vae,
            **kwargs,
        )
