import warnings
from pathlib import Path
from typing import Mapping, List, Union

import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

default_providers = [
    'CUDAExecutionProvider',
    'MIGraphXExecutionProvider',
    'ROCMExecutionProvider',
    'OpenVINOExecutionProvider',
    'CPUExecutionProvider',
]


def softmax(x) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


class OrientationDetection:
    all_angles = [0, 90, 180, 270]  # classes

    def __init__(self, model_path: Union[str, Path], providers: List[str] = None):
        if not str(model_path).lower().endswith(".onnx"):
            raise ValueError("model extention must be .onnx")
        if providers is None:
            providers = default_providers

        with warnings.catch_warnings():
            # UserWarning: Specified provider 'XXXXXXExecutionProvider' is not in available provider names.
            # Available providers: 'TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider'
            warnings.simplefilter("ignore", category=UserWarning)
            self.ort_session = onnxruntime.InferenceSession(model_path, providers=providers)

    @staticmethod
    def read_image(image_path: str) -> Image.Image:
        return Image.open(image_path)

    @staticmethod
    def to_array(image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        numpy_image: np.ndarray = transform(image).unsqueeze(0).numpy()
        return numpy_image

    def get_angles(self, numpy_image: np.ndarray) -> Mapping[int, float]:
        input_name = self.ort_session.get_inputs()[0].name
        ort_inputs = {input_name: numpy_image}
        ort_outs = self.ort_session.run(None, ort_inputs)
        logits = ort_outs[0]
        probabilities = softmax(logits)[0]
        angles = {angle: score for angle, score in zip(self.all_angles, probabilities)}
        return angles

    def get_angles_avg(self, numpy_image: np.ndarray) -> Mapping[int, float]:
        # create a dictionary to accumulate scores for each absolute orientation
        # (i.e. after “unrotating” the predictions).
        accumulated = {a: [] for a in self.all_angles}

        base_pred = None
        for rotation in self.all_angles:
            angles = self.get_angles(numpy_image)

            best_angle = max(angles, key=angles.get)
            if base_pred is None:
                # save the baseline best angle for the unrotated image
                base_pred = best_angle
            else:
                # calculate what the best angle should be after this rotation
                expected_angle = (base_pred + rotation) % 360
                assert best_angle == expected_angle

            # adjust each predicted angle to the absolute orientation (relative to the original image).
            # the conversion is: absolute_angle = (predicted_angle - rotation) mod 360.
            for pred_angle, score in angles.items():
                absolute_angle = (pred_angle - rotation) % 360
                accumulated[absolute_angle].append(score)

            # instead of using img.transpose(Image.Transpose.ROTATE_90)
            # rotate the transformed and proccessed image
            numpy_image = np.rot90(numpy_image, k=1, axes=(2, 3))

        result = dict()
        for angle in sorted(accumulated.keys()):
            overall_avg = sum(accumulated[angle]) / len(accumulated[angle])
            result[angle] = overall_avg
        return result


def main(model_path: str, image_path: str):
    od = OrientationDetection(model_path=model_path)
    img_array = od.to_array(od.read_image(image_path))
    angles = od.get_angles_avg(img_array)

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
