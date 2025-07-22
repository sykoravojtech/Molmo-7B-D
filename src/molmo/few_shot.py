"""Interactive few-shot chat with Molmo-7B-D."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

MODEL_ID: str = "allenai/Molmo-7B-D-0924"
IMAGE_PATH: Path = Path("data/images/diagram.png")
IMAGE_PATH1b: Path = Path("data/images/diagram2.png")

# ───────────────────────── Few‑shot examples ──────────────────────────── #

SHOT_1_USER: str = (
    "Analyze this first image of a circuit diagram and identify which electronic symbol classes "
    "are present. Rules: Only list symbols clearly visible, don't guess, exclude "
    "uncertain ones. Return as python list. Available classes: "
    '["terminal", "gnd", "vss", "voltage.dc", "voltage.ac", "voltage.battery", '
    '"resistor", "resistor.adjustable", "resistor.photo", '
    '"capacitor.unpolarized", "capacitor.polarized", "capacitor.adjustable", '
    '"inductor", "inductor.ferrite", "inductor.coupled", "transformer", '
    '"diode", "diode.light_emitting", "diode.thyrector", "diode.zener", '
    '"diac", "triac", "thyristor", "varistor", "transistor.bjt", '
    '"transistor.fet", "transistor.photo", "operational_amplifier", '
    '"operational_amplifier.schmitt_trigger", "optocoupler", '
    '"integrated_circuit", "integrated_circuit.ne555", '
    '"integrated_circuit.voltage_regulator", "xor", "and", "or", '
    '"not", "nand", "nor", "probe", "probe.current", "probe.voltage", '
    '"switch", "relay", "socket", "fuse", "speaker", "motor", '
    '"lamp", "microphone", "antenna", "crystal", "mechanical", '
    '"magnetic", "optical", "block"]'
)
SHOT_1_ASSISTANT: str = '["voltage.battery", "resistor", "diode.light_emitting"]'

SHOT_1b_USER: str = (
    "Analyze this second image of a circuit diagram and identify which electronic symbol classes "
    "are present. Rules: Only list symbols clearly visible, don't guess, exclude "
    "uncertain ones. Return as python list. Available classes: "
    '["terminal", "gnd", "vss", "voltage.dc", "voltage.ac", "voltage.battery", '
    '"resistor", "resistor.adjustable", "resistor.photo", '
    '"capacitor.unpolarized", "capacitor.polarized", "capacitor.adjustable", '
    '"inductor", "inductor.ferrite", "inductor.coupled", "transformer", '
    '"diode", "diode.light_emitting", "diode.thyrector", "diode.zener", '
    '"diac", "triac", "thyristor", "varistor", "transistor.bjt", '
    '"transistor.fet", "transistor.photo", "operational_amplifier", '
    '"operational_amplifier.schmitt_trigger", "optocoupler", '
    '"integrated_circuit", "integrated_circuit.ne555", '
    '"integrated_circuit.voltage_regulator", "xor", "and", "or", '
    '"not", "nand", "nor", "probe", "probe.current", "probe.voltage", '
    '"switch", "relay", "socket", "fuse", "speaker", "motor", '
    '"lamp", "microphone", "antenna", "crystal", "mechanical", '
    '"magnetic", "optical", "block"]'
)
SHOT_1b_ASSISTANT: str = '["resistor", "diode.zener", "gnd", "terminal"]'

# SHOT_2_USER: str = (
#     "You are a specialised circuit-diagram interpreter performing zero-shot "
#     "inference.\nGiven the image of an electrical circuit diagram, identify **all** "
#     'symbols of class "resistor".\n\n- Output exactly one <points> tag and *nothing '
#     "else*.\n- Inside that tag, list one coordinate pair per detected symbol, in "
#     'order:\n  x1="…" y1="…" x2="…" y2="…" x3="…" y3="…" … continuing as '
#     "needed.\n  Use the centre of each symbol's bounding box and the number being "
#     "the percentage of the image width and height.\n- If the image contains only "
#     "one symbol, output only x1 and y1; if two, add x2 y2; and so on.\n- Do **not** "
#     "invent coordinates only include pairs for symbols that truly exist."
#     # '\n\nExample for a single symbol:\n<points x1="41.0" y1="58.3" alt="resistor">resistor</points>\n\nExample '
#     # 'for four symbols:\n<points x1="12.1" y1="23.4" x2="45.0" y2="67.2" '
#     # 'x3="78.8" y3="11.9" x4="102.5" y4="53.0" alt="resistor">resistor</points>'
# )
SHOT_2_USER: str = (
    'You are a specialised circuit-diagram interpreter performing zero-shot inference. Given the image of an electrical circuit diagram, identify all symbols of class "resistor". For each symbol from that class, output one <points> tag and nothing else, in the exact format: <points x1="…" y1="…" x2="…" y2="…" x3="…" y3="…" alt="resistor">resistor</points> Use the coordinates of the center of the bounding box of each symbol. Do not include any additional text or tags.\nExample for 3 different locations: <points x1="20.8" y1="42.4" x2="21.0" y2="53.6" x3="22.3" y3=54.8 alt="resistor">resistor</points>'
)

SHOT_2_ASSISTANT: str = '<points x1="56.6" y1="24.4" alt="resistor">resistor</points>'

SYSTEM_PROMPT: str = "You are a precise and concise circuit-diagram interpreter."

# ─────────────────────── Model & processor helpers ──────────────────────── #


def load_model() -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    """Load Molmo-7B-D."""
    opts = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    return (
        AutoModelForCausalLM.from_pretrained(MODEL_ID, **opts),
        AutoProcessor.from_pretrained(MODEL_ID, **opts),
    )


def build_messages(
    query: str, user_image_path: str, shot: int = 0
) -> Tuple[List[dict[str, str]], List[Image.Image]]:
    """Return chat messages and matching image list."""
    shot_img = Image.open(IMAGE_PATH).convert("RGB")
    shot_img1b = Image.open(IMAGE_PATH1b).convert("RGB")
    user_img = Image.open(user_image_path).convert("RGB")
    messages = []
    images = []

    if shot == 1:
        messages.extend(
            [
                {"role": "user", "content": f"{SHOT_1_USER} <image>"},
                {"role": "assistant", "content": SHOT_1_ASSISTANT},
            ]
        )
        images.append(shot_img)
        messages.extend(
            [
                {"role": "user", "content": f"{SHOT_1b_USER} <image>"},
                {"role": "assistant", "content": SHOT_1b_ASSISTANT},
            ]
        )
        images.append(shot_img1b)
    elif shot == 2:
        messages.extend(
            [
                {"role": "user", "content": f"{SHOT_2_USER} <image>"},
                {"role": "assistant", "content": SHOT_2_ASSISTANT},
            ]
        )
        images.append(shot_img)

    # Add the actual user query and image
    messages.append({"role": "user", "content": f"{query} <image>"})
    images.append(user_img)

    return messages, images


def run_inference(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    query: str,
    user_image_path: str,
    shot: int = 0,
) -> str:
    """Generate Molmo reply for 'query' with the selected shot base."""
    messages, images = build_messages(query, user_image_path, shot)
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor.process(images=images, text=prompt)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # Convert image inputs to the same dtype as the model
    if "images" in inputs:
        inputs["images"] = inputs["images"].to(dtype=torch.bfloat16)

    with torch.autocast("cuda", torch.bfloat16):
        out = model.generate_from_batch(
            inputs,
            GenerationConfig(
                max_new_tokens=512, stop_strings="<|endoftext|>", do_sample=False
            ),
            tokenizer=processor.tokenizer,
        )

    generated_tokens = out[0, inputs["input_ids"].size(1) :]
    return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)


# ─────────────────────────────── Logging ─────────────────────────────── #


def setup_logging() -> str:
    """Setup logging file and return the log file path."""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Create log file with current datetime
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"few_shot_{current_time}.json"

    # Initialize the log file with empty list
    with open(log_file, "w") as f:
        json.dump([], f)

    return str(log_file)


def log_interaction(
    log_file: str,
    shot: int,
    image_path: str,
    human_prompt: str,
    assistant_response: str,
):
    """Log a single interaction to the JSON log file."""
    # Read existing log entries
    with open(log_file, "r") as f:
        log_entries = json.load(f)

    # Add new entry
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "shot_selection": shot,
        "image_path": image_path,
        "human_prompt": human_prompt,
        "assistant_response": assistant_response,
    }
    log_entries.append(new_entry)

    # Write back to file
    with open(log_file, "w") as f:
        json.dump(log_entries, f, indent=2)


# ─────────────────────────────── Main loop ─────────────────────────────── #


def main() -> None:
    """Start an endless prompt-reply loop until the user exits."""
    print("Loading Molmo-7B-D …")
    model, processor = load_model()

    # Setup logging
    log_file = setup_logging()
    print(f"Logging to: {log_file}")
    print("Model ready.  Type a prompt or 'exit' to quit.\n")

    while True:
        try:
            shot_input = input(
                "---\nShot selection (0=none, 1=symbol, 2=location): "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if shot_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        if not shot_input:
            print("not shot input")
            continue

        try:
            shot = int(shot_input)
            if shot not in [0, 1, 2]:
                print("Please enter 0, 1, or 2.")
                continue
        except ValueError:
            print("Please enter a valid number (0, 1, or 2).")
            continue

        try:
            user_image_path = input("---\nImage_Path: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if user_image_path.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        if not user_image_path:
            print("not user image path")
            continue

        if not os.path.isfile(user_image_path):
            print(
                f"File not found: {user_image_path!r}. Please enter a valid image path."
            )
            continue

        try:
            user_query = input("---\nHuman_Prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if user_query.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        if not user_query:
            print("not user query")
            continue

        print("---\nModel_Response: ", end="")
        response = run_inference(model, processor, user_query, user_image_path, shot)
        print(response)

        # Log the interaction
        log_interaction(log_file, shot, user_image_path, user_query, response)
        print(f"Logged interaction to {log_file}")


if __name__ == "__main__":
    main()