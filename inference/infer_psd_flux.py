import os
from pathlib import Path
import torch
from PIL import Image

from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig

# ========= Configuration =========
INPUT_TXT_DIR = Path("/PATH/TO/TESTSET")           # directory containing .txt prompt files
OUTPUT_ROOT   = Path("/PATH/TO/OUTPUT")             # output directory
LORA_PATH     = "/PATH/TO/LORA/psd_flux.safetensors"
LORA_ALPHA    = 1
BASE_SEED     = 0       # seed for pass i = BASE_SEED + i
DTYPE         = torch.bfloat16
DEVICE        = "cuda"

MODEL_CONFIGS = [
    ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
    ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
    ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
    ModelConfig(path="/PATH/TO/RGBA_VAE/diffusion_pytorch_model.safetensors"),
]

IMAGE_EXT  = ".png"
NUM_PASSES = 20     # number of generation passes per prompt (each uses a different seed)
# =================================


def ensure_dir(p: Path):
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)


def read_prompt(txt_path: Path) -> str:
    try:
        return txt_path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return txt_path.read_text(encoding="gb18030", errors="ignore").strip()


def find_groundtruth(img_dir: Path, stem: str) -> Path | None:
    for ext in ("png", "PNG"):
        p = img_dir / f"{stem}.{ext}"
        if p.exists():
            return p
    return None


def main():
    if not INPUT_TXT_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {INPUT_TXT_DIR}")

    ensure_dir(OUTPUT_ROOT)
    print(f"[INFO] Output directory: {OUTPUT_ROOT}")

    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=DTYPE,
        device=DEVICE,
        model_configs=MODEL_CONFIGS,
    )
    pipe.load_lora(pipe.dit, LORA_PATH, alpha=LORA_ALPHA)
    print("[INFO] Pipeline and LoRA loaded.")

    txt_files = sorted(INPUT_TXT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"[WARN] No .txt files found in: {INPUT_TXT_DIR}")
        return

    print(f"[INFO] Found {len(txt_files)} prompts. Running {NUM_PASSES} passes (seed {BASE_SEED} to {BASE_SEED + NUM_PASSES - 1}).")

    total_to_try = len(txt_files) * NUM_PASSES
    saved_count = skipped_count = fail_count = 0

    with torch.inference_mode():
        for pass_idx in range(NUM_PASSES):
            seed = BASE_SEED + pass_idx
            print(f"\n=== Pass {pass_idx + 1}/{NUM_PASSES} | seed={seed} ===")

            for txt_path in txt_files:
                stem   = txt_path.stem
                prompt = read_prompt(txt_path)
                if not prompt:
                    print(f"[SKIP] Empty prompt: {txt_path.name}")
                    skipped_count += 1
                    continue

                gt_path = find_groundtruth(INPUT_TXT_DIR, stem)
                gt_w = gt_h = None
                if gt_path is not None:
                    try:
                        with Image.open(gt_path) as gt_img:
                            gt_w, gt_h = gt_img.size
                        print(f"[INFO] Using GT size: {gt_w}x{gt_h}  ({gt_path.name})")
                    except Exception as e:
                        print(f"[WARN] Could not open GT image, using default size. Reason: {e}")

                out_dir   = OUTPUT_ROOT / stem
                ensure_dir(out_dir)

                save_name = f"{pass_idx:02d}_seed{seed}{IMAGE_EXT}"
                save_path = out_dir / save_name
                if save_path.exists():
                    print(f"[SKIP] Already exists: {save_path.name}")
                    skipped_count += 1
                    continue

                try:
                    if gt_w is not None and gt_h is not None:
                        try:
                            img = pipe(prompt=prompt, seed=seed, width=gt_w, height=gt_h)
                        except TypeError:
                            img = pipe(prompt=prompt, seed=seed)
                            if img.size != (gt_w, gt_h):
                                img = img.resize((gt_w, gt_h), Image.BICUBIC)
                    else:
                        img = pipe(prompt=prompt, seed=seed)

                    img.save(save_path)
                    saved_count += 1
                    print(f"[OK  ] Saved: {save_path}  (seed={seed})")
                except Exception as e:
                    print(f"[FAIL] {stem} [pass {pass_idx}, seed {seed}]: {e}")
                    fail_count += 1

    print(f"\n[DONE] Total attempted: {total_to_try}")
    print(f"       Saved:           {saved_count}")
    print(f"       Skipped:         {skipped_count}")
    print(f"       Failed:          {fail_count}")


if __name__ == "__main__":
    main()
