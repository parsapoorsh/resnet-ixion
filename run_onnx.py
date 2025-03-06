from pathlib import Path

import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def main(model_path: str, image_path: str):
    ort_session = onnxruntime.InferenceSession(model_path)
    img = Image.open(image_path).convert("RGB")

    tensor_image = transform(img).unsqueeze(0).numpy()

    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: tensor_image}

    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0]

    probabilities = softmax(logits)[0]

    all_angles = [0, 90, 180, 270]
    angles = {angle: score for angle, score in zip(all_angles, probabilities)}

    print("Orientation probabilities:")
    for angle, prob in angles.items():
        prob = prob * 100
        print(f"{angle}°: {prob:.2f}%")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("image_path", type=Path)
    args = parser.parse_args()

    main(args.model_path, args.image_path)
