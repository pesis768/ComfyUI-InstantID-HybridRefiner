import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HYBRID_NODES = new Set([
    "InstantIDKrea2FaceRefiner",
    "InstantIDMossFaceRefiner",
]);

const PRESET_PARAM_KEYS = [
    "ip_weight",
    "cn_strength",
    "kps_identity_morph",
    "crop_scale",
    "denoise",
    "steps",
    "cfg",
    "mask_expand",
    "mask_blur",
    "mouth_mask_keep",
    "nose_mask_keep",
    "hf_pore_transfer",
];

let cachedPresets = null;
let cachedBuiltinNames = new Set();

async function fetchPresetsFromServer() {
    try {
        const resp = await api.fetchApi("/instantid_hybrid/presets");
        if (resp.ok) {
            const data = await resp.json();
            if (data.status === "ok" && data.presets) {
                cachedPresets = data.presets;
                cachedBuiltinNames = new Set(data.builtin_names || []);
                return cachedPresets;
            }
        }
    } catch (err) {
        console.error("[InstantID Hybrid] Erro ao carregar presets:", err);
    }
    return cachedPresets || {};
}

function addFolderButton(node, folderWidgetName, boolWidgetName, buttonLabel, dialogTitle) {
    if (!node.widgets) return;
    const folderWidget = node.widgets.find((w) => w.name === folderWidgetName);
    if (!folderWidget) return;

    if (node.widgets.find((w) => w.name === buttonLabel)) return;

    const btn = node.addWidget("button", buttonLabel, null, async () => {
        const originalLabel = btn.label || buttonLabel;
        btn.label = "⏳ Abrindo seletor de pasta...";
        node.setDirtyCanvas(true, true);

        try {
            const resp = await api.fetchApi("/instantid_hybrid/select_folder", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    current_path: folderWidget.value || "",
                    title: dialogTitle,
                }),
            });

            if (resp.ok) {
                const data = await resp.json();
                if (data.status === "ok" && data.path) {
                    folderWidget.value = data.path;
                    if (folderWidget.callback) {
                        folderWidget.callback(data.path);
                    }
                    const boolWidget = node.widgets.find((w) => w.name === boolWidgetName);
                    if (boolWidget) {
                        boolWidget.value = true;
                        if (boolWidget.callback) {
                            boolWidget.callback(true);
                        }
                    }
                    const countInfo = typeof data.image_count === "number" ? ` (${data.image_count} fotos)` : "";
                    btn.label = `✅ Pasta selecionada${countInfo}`;
                    node.setDirtyCanvas(true, true);
                    setTimeout(() => {
                        btn.label = originalLabel;
                        node.setDirtyCanvas(true, true);
                    }, 3000);
                    return;
                }
            }
        } catch (err) {
            console.error("[InstantID Hybrid] Erro ao selecionar pasta:", err);
        }

        btn.label = originalLabel;
        node.setDirtyCanvas(true, true);
    });

    btn.serialize = false;
    btn.options = btn.options || {};
    btn.options.serialize = false;
}

function setupPresetControls(node) {
    if (!node.widgets) return;
    const presetWidget = node.widgets.find((w) => w.name === "preset_profile");
    if (!presetWidget) return;

    let isApplyingPreset = false;

    const refreshPresetDropdownValues = (presetsObj) => {
        if (!presetsObj) return;
        const keys = Object.keys(presetsObj);
        if (presetWidget.options && Array.isArray(presetWidget.options.values)) {
            presetWidget.options.values = keys;
        }
    };

    const applyPresetToSliders = async (presetName) => {
        if (!presetName || presetName.startsWith("manual")) return;
        const presets = cachedPresets || (await fetchPresetsFromServer());
        const vals = presets[presetName];
        if (!vals || typeof vals !== "object") return;

        isApplyingPreset = true;
        try {
            for (const key of PRESET_PARAM_KEYS) {
                if (vals[key] !== undefined) {
                    const targetW = node.widgets.find((w) => w.name === key);
                    if (targetW) {
                        targetW.value = vals[key];
                    }
                }
            }
            node.setDirtyCanvas(true, true);
        } finally {
            isApplyingPreset = false;
        }
    };

    const origPresetCallback = presetWidget.callback;
    presetWidget.callback = function (value) {
        const r = origPresetCallback ? origPresetCallback.apply(this, arguments) : undefined;
        applyPresetToSliders(value);
        return r;
    };

    for (const key of PRESET_PARAM_KEYS) {
        const paramW = node.widgets.find((w) => w.name === key);
        if (!paramW) continue;
        const origParamCb = paramW.callback;
        paramW.callback = function (value) {
            if (!isApplyingPreset && presetWidget.value && !presetWidget.value.startsWith("manual")) {
                presetWidget.value = "manual (usar controles abaixo)";
            }
            return origParamCb ? origParamCb.apply(this, arguments) : undefined;
        };
    }

    const saveBtnLabel = "💾 Salvar Configuração Atual como Preset...";
    if (!node.widgets.find((w) => w.name === saveBtnLabel)) {
        const saveBtn = node.addWidget("button", saveBtnLabel, null, async () => {
            const name = window.prompt(
                "Digite um nome para salvar o novo Preset com os valores atuais dos controles:",
                "meu_preset_personalizado"
            );
            if (!name || !name.trim()) return;

            const currentValues = {};
            for (const key of PRESET_PARAM_KEYS) {
                const w = node.widgets.find((item) => item.name === key);
                if (w !== undefined) {
                    currentValues[key] = w.value;
                }
            }

            try {
                const resp = await api.fetchApi("/instantid_hybrid/save_preset", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        name: name.trim(),
                        values: currentValues,
                    }),
                });
                const data = await resp.json();
                if (resp.ok && data.status === "ok") {
                    cachedPresets = data.presets;
                    cachedBuiltinNames = new Set(data.builtin_names || []);
                    refreshPresetDropdownValues(cachedPresets);
                    presetWidget.value = data.name;
                    const origLbl = saveBtn.label || saveBtnLabel;
                    saveBtn.label = `✅ Preset "${data.name}" salvo!`;
                    node.setDirtyCanvas(true, true);
                    setTimeout(() => {
                        saveBtn.label = origLbl;
                        node.setDirtyCanvas(true, true);
                    }, 3000);
                } else {
                    window.alert(data.message || "Erro ao salvar preset.");
                }
            } catch (err) {
                console.error("[InstantID Hybrid] Erro ao salvar preset:", err);
            }
        });
        saveBtn.serialize = false;
        saveBtn.options = saveBtn.options || {};
        saveBtn.options.serialize = false;
    }

    fetchPresetsFromServer().then((presets) => {
        refreshPresetDropdownValues(presets);
    });
}

function repairCorruptedWidgets(node) {
    if (!node || !node.widgets) return;
    const findW = (name) => node.widgets.find((w) => w.name === name);

    const checkCombo = (name, preferredFallback) => {
        const w = findW(name);
        if (!w) return;
        const allowed = w.options && Array.isArray(w.options.values) ? w.options.values : null;
        if (allowed && allowed.length > 0 && !allowed.includes(w.value)) {
            if (preferredFallback !== undefined && allowed.includes(preferredFallback)) {
                w.value = preferredFallback;
            } else {
                w.value = allowed[0];
            }
        }
    };

    const checkNumber = (name, min, max, fallback, isInt = false) => {
        const w = findW(name);
        if (!w) return;
        const num = Number(w.value);
        if (typeof w.value === "boolean" || typeof w.value === "string" && w.value.trim() === "" || Number.isNaN(num) || num < min || num > max) {
            w.value = fallback;
        } else {
            w.value = isInt ? Math.round(num) : num;
        }
    };

    const checkBool = (name, fallback) => {
        const w = findW(name);
        if (!w) return;
        if (typeof w.value !== "boolean") {
            w.value = fallback;
        }
    };

    checkBool("use_reference_folder", true);
    checkBool("use_reference_folder_2", false);
    checkCombo("target_gender", "all (todos)");
    checkCombo("target_position", "largest (maior / principal)");
    checkNumber("target_index", 1, 20, 1, true);
    checkCombo("preset_profile", "default (padrão equilibrado)");
    checkNumber("ip_weight", 0.0, 3.0, 0.95, false);
    checkNumber("cn_strength", 0.0, 3.0, 0.85, false);
    checkNumber("kps_identity_morph", 0.0, 1.0, 0.65, false);
    checkNumber("crop_scale", 1.2, 4.0, 2.0, false);
    checkNumber("denoise", 0.0, 1.0, 0.55, false);
    checkNumber("seed", 0, Number.MAX_SAFE_INTEGER, 500035378828582, true);
    checkCombo("control_after_generate", "fixed");
    checkNumber("steps", 1, 100, 15, true);
    checkNumber("cfg", 0.0, 30.0, 1.0, false);
    checkCombo("sampler_name", "lcm");
    checkCombo("scheduler", "exponential");
    checkCombo("mask_mode", "landmarks_106_forehead");
    checkNumber("mask_expand", -64, 128, 16, true);
    checkNumber("mask_blur", 1, 150, 32, true);
    checkNumber("mouth_mask_keep", 0.0, 1.0, 0.0, false);
    checkNumber("nose_mask_keep", 0.0, 1.0, 0.0, false);
    checkCombo("color_match", "LAB_skin_match");
    checkNumber("hf_pore_transfer", 0.0, 1.5, 0.25, false);

    // Krea 2 / MOSS specific polish controls
    const kreaDenoise = findW("krea2_polish_denoise");
    if (kreaDenoise) {
        if (typeof kreaDenoise.value !== "number" || Number.isNaN(kreaDenoise.value) || kreaDenoise.value > 0.45 || kreaDenoise.value < 0.0) {
            kreaDenoise.value = 0.0;
        }
    }
    checkNumber("krea2_polish_steps", 1, 50, 12, true);
    checkNumber("krea2_polish_cfg", 0.5, 10.0, 1.0, false);

    const mossDenoise = findW("moss_polish_denoise");
    if (mossDenoise) {
        if (typeof mossDenoise.value !== "number" || Number.isNaN(mossDenoise.value) || mossDenoise.value > 0.45 || mossDenoise.value < 0.0) {
            mossDenoise.value = 0.0;
        }
    }
    checkNumber("moss_polish_steps", 1, 50, 6, true);
    checkNumber("moss_polish_cfg", 0.5, 10.0, 1.0, false);
    checkBool("free_vram_before", true);

    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "InstantID.HybridFolderPicker",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!HYBRID_NODES.has(nodeData.name)) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

            addFolderButton(
                this,
                "reference_folder",
                "use_reference_folder",
                "📁 Selecionar Pasta de Referência 1...",
                "Selecione a pasta com fotos da pessoa (Referência 1)"
            );

            addFolderButton(
                this,
                "reference_folder_2",
                "use_reference_folder_2",
                "📁 Selecionar Pasta 2 (Casal / Homem)...",
                "Selecione a pasta com fotos da 2ª pessoa (Modo Casal: Homem)"
            );

            setupPresetControls(this);
            repairCorruptedWidgets(this);
            setTimeout(() => repairCorruptedWidgets(this), 100);

            return r;
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
            repairCorruptedWidgets(this);
            setTimeout(() => repairCorruptedWidgets(this), 100);
            return r;
        };
    },
});
