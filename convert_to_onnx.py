from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("pth_model_path", type=Path)
args = parser.parse_args()

pth_model_path = args.pth_model_path
output_name = pth_model_path.stem + ".onnx"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.resnet152(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
state_dict = torch.load(pth_model_path)
model.load_state_dict(state_dict)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    output_name,
    export_params=True,
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"},
                  "output": {0: "batch_size"}},
)

print(f"model has been converted to ONNX format and saved as {output_name}")
