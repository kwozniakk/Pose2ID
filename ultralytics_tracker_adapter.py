from typing import Iterable, List, Dict
try:
    from ultralytics.engine.results import Results  # type: ignore
except ImportError:  # allow running without ultralytics when using mocks
    class Results:  # minimal stand-in for type checking
        pass
import numpy as np
import random

def results_to_pose2id_inputs(results: Iterable[Results]) -> List[Dict]:
    """Convert Ultralytics YOLO tracker results to Pose2ID input format.

    Each item in the returned list has keys:
        - ``id``: track id
        - ``bbox``: [x1, y1, x2, y2]
        - ``conf``: confidence score
        - ``frame``: original image as ``numpy.ndarray``
        - ``frame_idx``: index of the frame in the input sequence
    """

    outputs = []
    for frame_idx, r in enumerate(results):
        im = r.orig_img
        boxes = r.boxes
        if boxes is None:
            continue
        for xyxy, track_id, conf in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.id.cpu().tolist(),
            boxes.conf.cpu().tolist(),
        ):
            outputs.append(
                {
                    "id": int(track_id),
                    "bbox": [float(x) for x in xyxy],
                    "conf": float(conf),
                    "frame": im,
                    "frame_idx": frame_idx,
                }
            )
    return outputs

def generate_mock_results(num_frames: int = 1, num_tracks: int = 3, size=(640, 480)) -> List[Results]:
    """Generate synthetic Ultralytics-style results for testing.

    Parameters
    ----------
    num_frames : int
        Number of video frames.
    num_tracks : int
        Number of unique object tracks per frame.
    size : tuple[int, int]
        Width and height of the generated images.
    """
    w, h = size

    class MockTensor:
        def __init__(self, data):
            self.data = data

        def cpu(self):
            return self

        def tolist(self):
            return self.data

    class MockBoxes:
        def __init__(self, xyxy, ids, confs):
            self.xyxy = MockTensor(xyxy)
            self.id = MockTensor(ids)
            self.conf = MockTensor(confs)

    class MockResult:
        def __init__(self, frame, boxes):
            self.orig_img = frame
            self.boxes = boxes

    results = []
    for _ in range(num_frames):
        frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        xyxy = []
        ids = []
        confs = []
        for tid in range(num_tracks):
            x1 = random.randint(0, w - 50)
            y1 = random.randint(0, h - 50)
            x2 = min(x1 + random.randint(30, 100), w)
            y2 = min(y1 + random.randint(60, 160), h)
            xyxy.append([x1, y1, x2, y2])
            ids.append(tid)
            confs.append(random.uniform(0.5, 1.0))
        boxes = MockBoxes(xyxy, ids, confs)
        results.append(MockResult(frame, boxes))
    return results
