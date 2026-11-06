# ComfyUI InstantID Hybrid Face Refiner

An advanced, high-fidelity **InstantID** face refiner suite for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), specifically designed to provide photo-realistic identity preservation with modern diffusion backends (**Krea 2**, **MOSS-Image SFT**, and **SDXL**).

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![ComfyUI Registry](https://img.shields.io/badge/Comfy_Registry-Available-green.svg)](https://comfyregistry.org)

---

## 🌟 Key Highlights & Features

### 1. 📁 Native File Explorer Folder Picker
- Integrated **"📁 Selecionar Pasta..."** button right inside the ComfyUI node.
- Automatically launches your operating system's native directory picker (Windows Explorer dialog) so you never have to copy & paste directory paths manually.
- Scans and caches multi-angle facial embeddings in RAM for instant batch generations.

### 2. 🎯 Multi-Face & Multi-Person Targeting
- **Filter by Gender**:
  - `all (todos)`: Processes every detected face.
  - `female_only (apenas mulher)`: Refines only female faces.
  - `male_only (apenas homem)`: Refines only male faces.
  - `couple_both (casal)`: Dual-reference mode — applies `reference_folder 1` to female faces and `reference_folder 2` to male faces in a single generation.
- **Filter by Spatial Position**:
  - `largest (maior / principal)`: Targets the main subject in the foreground.
  - `center (a do meio)`: Refines the center character in group photos.
  - `left_to_right`, `right_to_left`, `top_to_bottom`, `bottom_to_top`: Select any specific face index across ordered detections.

### 3. 👅 🚬 Anatomical Mask Protection (`mouth_mask_keep` & `nose_mask_keep`)
- Prevents InstantID from destroying characteristic facial expressions, open-mouth tongue poses, objects, and facial accessories:
  - **Tongue out / Open mouth**: Preserves the interior of the mouth and tongue from being smoothed out.
  - **Cigarettes / Straws / Vapes / Candy**: Keeps objects held in the lips or teeth intact.
  - **Piercings**: Preserves nose/septum piercings and jewelry with zero identity bleeding.

### 4. 🎛️ Instant Preset Profiles & Custom User Presets
Quickly switch between pre-tuned configurations or save your own:
- **`default (padrão equilibrado)`**: Balanced photo-realism and identity fidelity.
- **`high_fidelity (alta fidelidade facial sem distorções)`**: Maximum resemblance with strict anatomical preservation.
- **`strong_expression (expressão forte / língua para fora - sem máscara)`**: Soft keypoint morphing for dynamic laughing or open-mouth poses.
- **`mouth_object (cigarro / canudo / objeto na boca)`**: Optimized for characters with props between their lips.
- **`piercing_detail (piercing no nariz / sobrancelha)`**: Preserves jewelry and fine skin accents.
- **`mask_keep_mouth` & `mask_keep_nose_mouth`**: Mask-blended protection modes.
- **💾 Salvar Configuração Atual como Preset...**: Save any slider combination with a custom name stored in `user_presets.json`.

### 5. 🔬 Hybrid Polish & Pore Detail Transfer
- **MOSS-Image SFT & Krea 2 Polish**: Built-in 2nd-stage micro-denoise polish that re-melds the face crop back into the full canvas with perfect seamless lighting and skin tone matching (`LAB_skin_match`).
- **High-Frequency Pore Transfer (`hf_pore_transfer`)**: Extracts and re-injects microscopic skin pore textures to prevent "plastic skin" or over-smoothed looks.

---

## 📦 Nodes Included

| Node Name | Display Name | Purpose |
|---|---|---|
| `InstantIDKrea2FaceRefiner` | **InstantID Face Refiner (Krea 2 / Hybrid)** | Full InstantID pipeline + Krea 2 micro-polish & pore transfer. |
| `InstantIDMossFaceRefiner` | **InstantID Face Refiner (MOSS-Image SFT)** | Full InstantID pipeline + MOSS-Image SFT micro-polish & pore transfer. |
| `InstantIDModelLoader` | **Load InstantID Model** | Loads InstantID IP-Adapter checkpoint (`ip-adapter.bin`). |
| `InstantIDFaceAnalysis` | **InstantID Face Analysis** | Loads InsightFace (`antelopev2` / `buffalo_l`). |
| `ApplyInstantID` | **Apply InstantID** | Core InstantID identity adapter applier. |
| `ApplyInstantIDAdvanced` | **Apply InstantID Advanced** | Advanced step-range & weight scheduling. |
| `FaceKeypointsPreprocessor`| **Face Keypoints Preprocessor** | 5-point facial landmark extractor. |
| `InstantIDAttentionPatch` | **InstantID Patch Attention** | Cross-attention replacement patch. |
| `ApplyInstantIDControlNet` | **InstantID Apply ControlNet** | InstantID ControlNet conditioning injector. |

---

## 🚀 Installation

### Option 1: ComfyUI Manager (Recommended)
1. Open ComfyUI and open the **ComfyUI Manager**.
2. Click **Install Custom Nodes**.
3. Search for **`InstantID Hybrid Face Refiner`** and click **Install**.
4. Restart ComfyUI.

*(If you open a workflow that already uses these nodes, simply click **"Install Missing Custom Nodes"** in ComfyUI Manager!)*

### Option 2: Git Clone
Open your terminal in `ComfyUI/custom_nodes/` and run:
```bash
git clone https://github.com/pesis768/ComfyUI-InstantID-HybridRefiner.git
```
Then install the dependencies:
```bash
cd ComfyUI-InstantID-HybridRefiner
pip install -r requirements.txt
```
Restart ComfyUI.

---

## 📥 Required Models

Place the required model files in their respective ComfyUI directories:

1. **InsightFace Models** (`ComfyUI/models/insightface/models/antelopev2/`):
   - Download the `antelopev2` model package (unzip so `.onnx` files are inside `models/insightface/models/antelopev2/`).
2. **InstantID Adapter** (`ComfyUI/models/instantid/`):
   - `ip-adapter.bin` from [InstantX/InstantID](https://huggingface.co/InstantX/InstantID).
3. **InstantID ControlNet** (`ComfyUI/models/controlnet/`):
   - `diffusion_pytorch_model.safetensors` (rename to `instantid-controlnet.safetensors`).
4. **SDXL Checkpoint** (`ComfyUI/models/checkpoints/`):
   - Any SDXL photorealistic checkpoint (e.g., *Juggernaut XL*, *RealVisXL*, *ZavyChromaXL*).
5. *(Optional for Hybrid Workflows)*:
   - **Krea 2 Checkpoint**: `kreros1.0_nvfp4.safetensors` or fp8 version.
   - **MOSS-Image SFT**: `moss_image_sft_turbo...safetensors` + `flux2-vae.safetensors` + `qwen3.5_9b_w4a8_gptq.safetensors`.

---

## 🎨 Recommended Workflows on Civitai

This suite powers the following official workflows published on Civitai:
1. **MOSS-Image SFT + Krea 2 Refiner + InstantID Hybrid (3-Pass)**: The ultimate pipeline combining MOSS composition, Krea 2 photo-refinement, and InstantID identity fidelity.
2. **MOSS-Image SFT + InstantID Hybrid**: Fast single-stage MOSS generation with InstantID face swap/refiner.
3. **Krea 2 + InstantID Hybrid**: Ultra-crisp photorealistic rendering with Krea 2 and InstantID.

---

## ⚙️ Key Parameter Tips

| Parameter | Recommended Value | Explanation |
|---|---|---|
| `ip_weight` | `0.90 - 1.15` | Identity weight. Increase for stronger resemblance; lower if face structure distorts. |
| `cn_strength` | `0.70 - 0.85` | Keypoint control strength. Keeps the facial structure anchored. |
| `kps_identity_morph` | `0.65 - 0.90` | Blend between base image expression and reference facial proportions. Lower for strong expressions (laughing/yawning). |
| `crop_scale` | `1.90 - 2.20` | How much context around the face is cropped for refinement. |
| `denoise` | `0.55 - 0.68` | Refinement denoise level. `0.64` is the sweet spot for clean identity transfer. |
| `mouth_mask_keep` | `0.50 - 1.00` | Turn up when the character has a tongue out, cigarette, straw, or open mouth. |
| `nose_mask_keep` | `0.50 - 1.00` | Turn up to preserve septum and nostril piercings. |
| `hf_pore_transfer` | `0.25 - 0.40` | Injects natural skin pores and avoids plastic textures. |

---

## 📄 License
This project is licensed under the [Apache License 2.0](LICENSE).
Built upon the foundations of [InstantX/InstantID](https://github.com/InstantX-Research/InstantID) and ComfyUI.
