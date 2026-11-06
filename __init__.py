from .hybrid_refiner import InstantIDKrea2FaceRefiner, InstantIDMossFaceRefiner
from .InstantID import (
    InstantIDModelLoader,
    InstantIDFaceAnalysis,
    ApplyInstantID,
    ApplyInstantIDAdvanced,
    FaceKeypointsPreprocessor,
    InstantIDAttentionPatch,
    ApplyInstantIDControlNet,
)

NODE_CLASS_MAPPINGS = {
    "InstantIDKrea2FaceRefiner": InstantIDKrea2FaceRefiner,
    "InstantIDMossFaceRefiner": InstantIDMossFaceRefiner,
    "InstantIDModelLoader": InstantIDModelLoader,
    "InstantIDFaceAnalysis": InstantIDFaceAnalysis,
    "ApplyInstantID": ApplyInstantID,
    "ApplyInstantIDAdvanced": ApplyInstantIDAdvanced,
    "FaceKeypointsPreprocessor": FaceKeypointsPreprocessor,
    "InstantIDAttentionPatch": InstantIDAttentionPatch,
    "ApplyInstantIDControlNet": ApplyInstantIDControlNet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantIDKrea2FaceRefiner": "InstantID Face Refiner (Krea 2 / Hybrid)",
    "InstantIDMossFaceRefiner": "InstantID Face Refiner (MOSS-Image SFT)",
    "InstantIDModelLoader": "Load InstantID Model",
    "InstantIDFaceAnalysis": "InstantID Face Analysis",
    "ApplyInstantID": "Apply InstantID",
    "ApplyInstantIDAdvanced": "Apply InstantID Advanced",
    "FaceKeypointsPreprocessor": "Face Keypoints Preprocessor",
    "InstantIDAttentionPatch": "InstantID Patch Attention",
    "ApplyInstantIDControlNet": "InstantID Apply ControlNet",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
