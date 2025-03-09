import sys
from argparse import ArgumentParser
from collections import deque
from pathlib import Path

from PIL import Image
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap, QTransform, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget, QGraphicsDropShadowEffect)

from run_onnx import OrientationDetection


# threshold to skip
threshold = 98.0 / 100


def callback(angle: int, score: float, image_path: str):
    print(str(angle).ljust(3), str(round(score * 100, 2)).ljust(2), image_path)


def recursive_iterdir(path: Path):
    path = Path(path)
    for i in path.iterdir():
        if i.is_dir():
            yield from recursive_iterdir(i)
        yield i


class Worker(QObject):
    image_found = pyqtSignal(Path, dict)
    finished = pyqtSignal()
    interrupted = pyqtSignal()

    def __init__(self, model_path: Path, directory_path: Path):
        super().__init__()
        self.model_path = model_path
        self.directory_path = directory_path
        self._is_running = True

    def run(self):
        od = OrientationDetection(model_path=self.model_path)
        Image.init()

        for image_path in recursive_iterdir(self.directory_path):
            if image_path.suffix.lower() not in Image.EXTENSION:
                continue
            if not self._is_running:
                break

            img_array = od.to_array(od.read_image(image_path))
            try:
                angles = od.get_angles_avg(img_array)
            except AssertionError:
                continue
            best_angle = max(angles, key=angles.get)
            best_score = angles[best_angle]

            if best_score > threshold:
                callback(best_angle, best_score, image_path)
                continue

            if best_angle != 0:
                self.image_found.emit(image_path, angles)
        self.finished.emit()

    @pyqtSlot()
    def stop(self):
        self._is_running = False


class ImageLabeler(QMainWindow):
    def __init__(self, worker: Worker, thread: QThread):
        super().__init__()
        self.processing_ui = False
        self.setWindowTitle("Image Labeler")

        # Thread management
        self.worker_thread = thread
        self.worker = worker

        # Image queue and worker status
        self.image_queue = deque()
        self.worker_finished = False

        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout()

        # Label to display the image
        self.image_label = QLabel("Image will appear here")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.image_label)

        # Buttons layout
        self.buttons_layout = QHBoxLayout()
        self.main_layout.addLayout(self.buttons_layout)

        # Initial UI update
        self.main_widget.setLayout(self.main_layout)
        self.update_image()

    def on_worker_finished(self):
        self.worker_finished = True
        self.update_image()

    def add_image_to_queue(self, image_path: Path, oinfo: dict):
        prev_len = len(self.image_queue)
        self.image_queue.append((image_path, oinfo))

        # trigger update if queue was empty and UI is in waiting state
        if prev_len == 0 and self.processing_ui:
            self.update_image()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_W:
            self.trigger_button(180)
        elif key == Qt.Key_D:
            self.trigger_button(90)
        elif key == Qt.Key_S:
            self.trigger_button(0)
        elif key == Qt.Key_A:
            self.trigger_button(270)
        elif key == Qt.Key_Space:
            self.trigger_button("BAD")
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            self.trigger_button("BEST")
        else:
            super().keyPressEvent(event)

    def trigger_button(self, btn_type):
        for i in range(self.buttons_layout.count()):
            widget: QPushButton = self.buttons_layout.itemAt(i).widget()
            if widget and (getattr(widget, 'angle', None) == btn_type or
                           getattr(widget, 'is_best', None) == btn_type):
                widget.click()
                break

    def recreate_buttons(self, oinfo, image_path=None):
        buttons = [
            ("⬇️", "Down", 0),
            ("⬆️", "Up", 180),
            ("⬅️", "Left", 270),
            ("➡️", "Right", 90),
            ("❌", "Bad", "BAD")
        ]

        # remove all buttons
        while self.buttons_layout.count():
            widget = self.buttons_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()

        best_angle = max(oinfo, key=oinfo.get)

        for btn_text, action, angle in buttons:
            btn_text = f"{btn_text} {action}"
            score = 0

            if angle in oinfo.keys():
                score = oinfo[angle]
                btn_text += f"\n{angle}°: {score * 100:.2f}%"

            btn = QPushButton(btn_text)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.angle = angle

            if angle == best_angle:
                shadow = QGraphicsDropShadowEffect()
                shadow.setBlurRadius(20)
                shadow.setColor(Qt.white)
                shadow.setOffset(0)
                btn.setGraphicsEffect(shadow)
                btn.is_best = "BEST"

            btn.clicked.connect(lambda checked, a=angle, s=score: self.process_action(a, s, image_path=image_path))
            self.buttons_layout.addWidget(btn)

        self.buttons_layout.update()

    def update_image(self):
        if not self.image_queue:
            if self.worker_finished:
                self.image_label.setText("No more images.")
                self.disable_buttons()
            else:
                self.image_label.setText("Processing images...")
                self.disable_buttons()
                self.processing_ui = True
            return
        self.processing_ui = False

        image_path, oinfo = self.image_queue.popleft()

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.image_label.setText(f"Cannot load image: {image_path}")
            return

        self.buttons_layout.invalidate()
        self.recreate_buttons(oinfo, image_path=image_path)

        best_angle = max(oinfo, key=oinfo.get)

        transform = QTransform().rotate(best_angle)
        rotated_pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)

        target_size = 640
        scaled_original = pixmap.scaled(target_size, target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scaled_rotated = rotated_pixmap.scaled(target_size, target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        line_width = 5
        total_width = scaled_original.width() + line_width + scaled_rotated.width()
        total_height = max(scaled_original.height(), scaled_rotated.height())
        composite = QPixmap(total_width, total_height)
        composite.fill(Qt.transparent)

        painter = QPainter(composite)
        painter.drawPixmap(0, 0, scaled_original)
        pen = QPen(Qt.black)
        pen.setWidth(line_width)
        painter.setPen(pen)
        x_line = scaled_original.width() + line_width // 2
        painter.drawLine(x_line, 0, x_line, total_height)
        painter.drawPixmap(scaled_original.width() + line_width, 0, scaled_rotated)
        painter.end()

        self.image_label.setPixmap(composite)

    def process_action(self, angle, score, image_path):
        callback(angle, score, image_path)
        self.update_image()

    def disable_buttons(self):
        for i in range(self.buttons_layout.count()):
            widget = self.buttons_layout.itemAt(i).widget()
            widget.setEnabled(False)


def main(model_path: Path, directory_path: Path):
    app = QApplication(sys.argv)

    # Setup worker thread
    worker = Worker(model_path, directory_path)
    thread = QThread()
    worker.moveToThread(thread)

    window = ImageLabeler(worker=worker, thread=thread)

    thread.started.connect(worker.run)
    thread.finished.connect(thread.deleteLater)
    worker.finished.connect(thread.quit)
    worker.interrupted.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.image_found.connect(window.add_image_to_queue)
    worker.finished.connect(window.on_worker_finished)

    thread.start()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("directory_path", type=Path)
    args = parser.parse_args()

    main(**vars(args))
