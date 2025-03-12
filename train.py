#!/usr/bin/env python3
import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


# Hyperparameters and settings
num_epochs = 4
batch_size = 26
learning_rate = 0.001

train_folder = ["coco/train2017", "coco/test2017"]
val_folder = ["coco/val2017"]
allowed_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# DO NOT CHANGE, unless you know what you are doing
# wget https://download.pytorch.org/models/resnet152-f82ba261.pth
checkpoint_path = "resnet152-f82ba261.pth"
classes = [0, 90, 180, 270]  # Four rotation classes: 0°, 90°, 180°, 270°
num_workers = os.cpu_count()
num_classes = len(classes)
epoches_done = 0

transform_train = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.RandomResizedCrop(224),
    # transforms.Resize((224, 224)),

    transforms.RandomAutocontrast(),  # F.autocontrast,
    # transforms.RandomEqualize(),
    # v2.JPEG((50, 100)),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def recursive_iterdir(path: Path):
    path = Path(path)
    for i in path.iterdir():
        if i.is_dir():
            yield from recursive_iterdir(i)
        yield i


class RotatedDataset(Dataset):
    def __init__(
        self, folders: list[str], transform=None, limit: int = 0, offset: int = 0
    ):
        self.folders = [folders] if isinstance(folders, str) else folders
        self.transform = transform
        self.limit = limit
        self.offset = offset
        self.image_paths = []

        # recursive iterate through directories
        for f in self.folders:
            for i in recursive_iterdir(f):
                # image.PnG -> .PnG -> .png
                if not i.suffix.lower().endswith(allowed_extensions):
                    continue
                self.image_paths.append(str(i))

    def __len__(self):
        total = len(self.image_paths) * num_classes
        if self.limit:
            total = self.limit
        return total - self.offset

    # the lst[0] returns the original image
    # but the lst[1], lst[2], lst[3] returns rotations of the original image
    # so in this way, each image would be x4'rd
    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        idx = idx + self.offset
        if idx >= len(self.image_paths) * num_classes:
            raise IndexError

        # the index of the original image in self.image_paths
        index = idx // num_classes

        # the index of the class.
        # 0 = 0°, 1 = 90°, 2 = 180°, 3 = 270°
        label = idx % num_classes

        img_path = self.image_paths[index]
        # noinspection PyBroadException
        try:
            image = Image.open(img_path).convert("RGB")
        # some time some images are damaged, or not readable
        # just return None, and they will be filtered by the collate_fn function
        except Exception:
            return None

        # randomly horizontaly flip the image, like transforms.RandomHorizontalFlip, p = 0.5
        if torch.rand(1).item() > 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        method = [
            None,
            Image.Transpose.ROTATE_90,
            Image.Transpose.ROTATE_180,
            Image.Transpose.ROTATE_270,
        ][label]
        if method is not None:
            image = image.transpose(method)

        if self.transform:
            image = self.transform(image)

        return image, label


train_dataset = RotatedDataset(train_folder, transform=transform_train)
val_dataset = RotatedDataset(val_folder, transform=transform_val)


# filter bad / damaged images
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=collate_fn,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.resnet152(weights=None)
# load pretrained weights (skipping the final fc layer)
state_dict = torch.load(checkpoint_path)
model_dict = model.state_dict()
pretrained_dict = {
    k: v for k, v in state_dict.items() if k in model_dict and "fc" not in k
}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# get the number of input features for the existing fc layer (model.fc.in_features)
# replace the final fully connected layer with a new one that outputs 4 classes
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

for epoch in range(epoches_done + 1, num_epochs + 1):
    print(f"Epoch [{epoch}/{num_epochs}] Starting...")

    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
        images: torch.Tensor = images.to(device)
        labels: torch.Tensor = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(dim=0)

    torch.save(model.state_dict(), f"resnet152_ixion_e{epoch}.pth")

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # cleanup VRAM
    torch.cuda.empty_cache()

    # a full validation of the val_dataset
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images: torch.Tensor = images.to(device)
            labels: torch.Tensor = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(dim=0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_acc * 100:.4f}%"
    )
    epoches_done += 1
