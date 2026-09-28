import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 512

def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-image demo for image forgery localization."
    )
    parser.add_argument("--image", required=True, help="Path to one input image.")
    parser.add_argument("--ckpt", required=True, help="Path to the trained checkpoint.")
    parser.add_argument("--output", default="./demo_mask.png", help="Output mask path.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-prob", action="store_true", help="Also save probability map.")
    return parser.parse_args()


def build_model():
    # A PyTorch checkpoint stores weights, so the compatible model definition
    # must still be available. ``cross_main`` is a compact inference wrapper.
    from cross_main import cross_main

    return cross_main()


def load_checkpoint(model, ckpt_path):
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint must be a state dict or contain a 'model' state dict.")

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)


def preprocess_image(image_path, input_size):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((input_size, input_size), Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def main():
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA is unavailable. Falling back to CPU.")
        device = "cpu"

    image_path = Path(args.image)
    ckpt_path = Path(args.ckpt)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model()
    load_checkpoint(model, str(ckpt_path))
    model.set_training_stage("stage3")
    model.to(device).eval()

    image_tensor = preprocess_image(image_path, INPUT_SIZE).to(device)
    with torch.no_grad():
        pred_prob = model.predict(image_tensor)

    prob = pred_prob[0, 0].detach().cpu().numpy().astype(np.float32)
    prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
    prob = np.clip(prob, 0.0, 1.0)
    mask = (prob >= args.threshold).astype(np.uint8) * 255

    Image.fromarray(mask).save(output_path)
    if args.save_prob:
        prob_path = output_path.with_name(output_path.stem + "_prob.png")
        Image.fromarray((prob * 255.0).astype(np.uint8)).save(prob_path)

    print(f"Saved mask: {output_path}")


if __name__ == "__main__":
    main()
