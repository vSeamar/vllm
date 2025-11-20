# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import math
from abc import abstractmethod
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from vllm import envs
from vllm.logger import init_logger
from vllm.utils.registry import ExtensionManager

from .base import MediaIO
from .image import ImageMediaIO

logger = init_logger(__name__)


def resize_video(frames: npt.NDArray, size: tuple[int, int]) -> npt.NDArray:
    num_frames, _, _, channels = frames.shape
    new_height, new_width = size
    resized_frames = np.empty(
        (num_frames, new_height, new_width, channels), dtype=frames.dtype
    )
    # lazy import cv2 to avoid bothering users who only use text models
    import cv2

    for i, frame in enumerate(frames):
        resized_frame = cv2.resize(frame, (new_width, new_height))
        resized_frames[i] = resized_frame
    return resized_frames


def rescale_video_size(frames: npt.NDArray, size_factor: float) -> npt.NDArray:
    _, height, width, _ = frames.shape
    new_height = int(height * size_factor)
    new_width = int(width * size_factor)

    return resize_video(frames, (new_height, new_width))


def sample_frames_from_video(frames: npt.NDArray, num_frames: int) -> npt.NDArray:
    total_frames = frames.shape[0]
    if num_frames == -1:
        return frames

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    sampled_frames = frames[frame_indices, ...]
    return sampled_frames


class VideoLoader:
    @classmethod
    @abstractmethod
    def load_bytes(
        cls, data: bytes, num_frames: int = -1, **kwargs
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        raise NotImplementedError

    @classmethod
    def _load_frames_with_seeking(
        cls,
        cap,
        frame_indices: list[int],
        recovery_offset: int,
    ) -> tuple[npt.NDArray, list[int], dict[int, int]]:
        """
        Load frames using seeking (random access) with optional recovery.

        This method jumps directly to each target frame using seeking,
        which allows bypassing corrupted frames that would cause early
        termination in sequential grab/retrieve approaches.

        Args:
            cap: VideoCapture object (must support seeking)
            frame_indices: List of frame indices to load
            recovery_offset: Search radius for recovery. If a frame fails,
                           tries nearby frames at offsets -1,+1,-2,+2,...
                           up to ±recovery_offset

        Returns:
            frames: Numpy array of loaded frames
            loaded_indices: List of frame indices that were successfully loaded
            recovered_frames: Dict mapping original_idx -> actual_idx for
                            recovered frames
        """
        import cv2

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_list = []
        loaded_indices = []
        recovered_frames = {}

        for target_idx in frame_indices:
            # Seek to target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()

            if not ret and recovery_offset > 0:
                # Attempt recovery by seeking to nearby frames
                logger.warning(
                    "Frame %d failed, attempting recovery (offset=%d)",
                    target_idx,
                    recovery_offset,
                )

                # Try offsets: -1, +1, -2, +2, ..., -offset, +offset
                for i in range(1, recovery_offset + 1):
                    for direction in [-i, i]:
                        recovery_idx = target_idx + direction

                        # Boundary checks
                        if recovery_idx < 0 or recovery_idx >= total_frames:
                            continue
                        # Don't use adjacent sampled frames
                        if recovery_idx in frame_indices:
                            continue

                        cap.set(cv2.CAP_PROP_POS_FRAMES, recovery_idx)
                        ret, frame = cap.read()

                        if ret:
                            logger.info(
                                "Recovered frame %d using frame %d (offset %+d)",
                                target_idx,
                                recovery_idx,
                                direction,
                            )
                            recovered_frames[target_idx] = recovery_idx
                            break

                    if ret:
                        break

            if ret:
                frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                loaded_indices.append(target_idx)
            else:
                if recovery_offset > 0:
                    logger.warning(
                        "Frame %d recovery failed after trying %d frames",
                        target_idx,
                        recovery_offset * 2,
                    )
                else:
                    logger.warning("Frame %d failed to load", target_idx)

        # Convert to numpy array
        if frames_list:
            frames = np.stack(frames_list)
        else:
            frames = np.empty((0, height, width, 3), dtype=np.uint8)

        return frames, loaded_indices, recovered_frames


VIDEO_LOADER_REGISTRY = ExtensionManager()


@VIDEO_LOADER_REGISTRY.register("opencv")
class OpenCVVideoBackend(VideoLoader):
    def get_cv2_video_api(self):
        import cv2.videoio_registry as vr

        api_pref = None
        for backend in vr.getStreamBufferedBackends():
            if not vr.hasBackend(backend):
                continue
            if not vr.isBackendBuiltIn(backend):
                _, abi, api = vr.getStreamBufferedBackendPluginVersion(backend)
                if abi < 1 or (abi == 1 and api < 2):
                    continue
            api_pref = backend
            break
        return api_pref

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = -1,
        recovery_offset: int = 0,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        import cv2

        backend = cls().get_cv2_video_api()
        cap = cv2.VideoCapture(BytesIO(data), backend, [])
        if not cap.isOpened():
            raise ValueError("Could not open video stream")

        total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames_num / original_fps if original_fps > 0 else 0

        # resample video to target num_frames and fps
        # - the minimum of the two will be used
        num_frames_to_sample = total_frames_num
        if num_frames > 0:
            num_frames_to_sample = min(num_frames, total_frames_num)
        if fps > 0:
            num_frames_to_sample = min(num_frames_to_sample, math.floor(duration * fps))
        num_frames_to_sample = max(1, num_frames_to_sample)  # at least one sample

        if num_frames_to_sample == total_frames_num:
            frame_idx = list(range(0, num_frames_to_sample))
        else:
            uniform_sampled_frames = np.linspace(
                0, total_frames_num - 1, num_frames_to_sample, dtype=int
            )
            frame_idx = uniform_sampled_frames.tolist()

        # Use seeking-based loading with recovery if enabled
        if recovery_offset > 0:
            logger.info("Frame recovery enabled (offset=%d)", recovery_offset)
            frames, loaded_indices, recovered_frames = cls._load_frames_with_seeking(
                cap, frame_idx, recovery_offset=recovery_offset
            )

            if recovered_frames:
                logger.info(
                    "Loaded %d/%d frames: %d original, %d recovered. Mappings: %s",
                    len(loaded_indices),
                    len(frame_idx),
                    len(loaded_indices) - len(recovered_frames),
                    len(recovered_frames),
                    recovered_frames,
                )

            frame_idx = loaded_indices
        else:
            # Use existing sequential grab/retrieve logic
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = np.empty((len(frame_idx), height, width, 3), dtype=np.uint8)

            i = 0
            for idx in range(max(frame_idx) + 1):
                ok = cap.grab()
                if not ok:
                    break
                if idx in frame_idx:
                    ret, frame = cap.retrieve()
                    if ret:
                        frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        i += 1

            # Replace assertion with warning for graceful degradation
            if i < num_frames_to_sample:
                logger.warning(
                    "Expected reading %d frames, but only loaded %d frames from video. "
                    "Continuing with partial data.",
                    num_frames_to_sample,
                    i,
                )
                frames = frames[:i]
                frame_idx = frame_idx[:i]

        # Use transformers transformers.video_utils.VideoMetadata format
        # NOTE(Isotr0py): For models like Qwen3-VL/GLM4.5V, this metadata
        # can cause incorrect timestamp calculation without num_frames=-1.
        metadata = {
            "total_num_frames": total_frames_num,
            "fps": original_fps,
            "duration": duration,
            "video_backend": "opencv",
            "frames_indices": list(frame_idx),
            # extra field used to control hf processor's video
            # sampling behavior
            "do_sample_frames": len(frame_idx) == total_frames_num,
        }

        return frames, metadata


@VIDEO_LOADER_REGISTRY.register("opencv_dynamic")
class OpenCVDynamicVideoBackend(OpenCVVideoBackend):
    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = 2,
        max_duration: int = 300,
        recovery_offset: int = 0,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        import cv2

        backend = cls().get_cv2_video_api()
        cap = cv2.VideoCapture(BytesIO(data), backend, [])
        if not cap.isOpened():
            raise ValueError("Could not open video stream")

        total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames_num / original_fps if original_fps > 0 else 0

        # resample video to target num_frames
        max_frame_idx = total_frames_num - 1
        duration = duration or round(max_frame_idx / original_fps) + 1

        # Refer to:
        # https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/glm4v/video_processing_glm4v.py#L103-L140
        frame_indices: range | list[int]
        if duration <= max_duration:
            n = int(math.floor(duration * fps))
            frame_indices = sorted(
                {
                    min(max_frame_idx, int(math.ceil(i * original_fps / fps)))
                    for i in range(n)
                }
            )
        else:
            num_samples = int(max_duration * fps)
            if num_samples >= total_frames_num:
                frame_indices = range(total_frames_num)
            else:
                target_seconds = np.linspace(0, duration, num_samples, endpoint=True)
                frame_indices = sorted(
                    {
                        min(max_frame_idx, int(math.ceil(t * original_fps)))
                        for t in target_seconds
                    }
                )

        # Convert to list for consistency
        frame_indices = list(frame_indices)

        # Use seeking-based loading with recovery if enabled
        if recovery_offset > 0:
            logger.info("Frame recovery enabled (offset=%d)", recovery_offset)
            frames, loaded_indices, recovered_frames = cls._load_frames_with_seeking(
                cap, frame_indices, recovery_offset=recovery_offset
            )

            if recovered_frames:
                logger.info(
                    "Loaded %d/%d frames: %d original, %d recovered. Mappings: %s",
                    len(loaded_indices),
                    len(frame_indices),
                    len(loaded_indices) - len(recovered_frames),
                    len(recovered_frames),
                    recovered_frames,
                )

            frame_indices = loaded_indices
        else:
            # Use existing sequential grab/retrieve logic
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = np.empty((len(frame_indices), height, width, 3), dtype=np.uint8)

            i = 0
            for idx in range(total_frames_num):
                ok = cap.grab()
                if not ok:
                    break
                if idx in frame_indices:
                    ret, frame = cap.retrieve()
                    if ret:
                        frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        i += 1

            # Replace assertion with warning for graceful degradation
            if i < len(frame_indices):
                logger.warning(
                    "Expected reading %d frames, but only loaded %d frames from video. "
                    "Continuing with partial data.",
                    len(frame_indices),
                    i,
                )
                frames = frames[:i]
                frame_indices = frame_indices[:i]

        # Use transformers transformers.video_utils.VideoMetadata format
        metadata = {
            "total_num_frames": total_frames_num,
            "fps": original_fps,
            "duration": duration,
            "video_backend": "opencv_dynamic",
            "frames_indices": list(frame_indices),
            "do_sample_frames": False,
        }

        return frames, metadata


class VideoMediaIO(MediaIO[npt.NDArray]):
    def __init__(
        self,
        image_io: ImageMediaIO,
        num_frames: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()

        self.image_io = image_io
        self.num_frames = num_frames
        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality.
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.
        self.kwargs = kwargs
        video_loader_backend = envs.VLLM_VIDEO_LOADER_BACKEND
        self.video_loader = VIDEO_LOADER_REGISTRY.load(video_loader_backend)

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, dict[str, Any]]:
        return self.video_loader.load_bytes(
            data, num_frames=self.num_frames, **self.kwargs
        )

    def load_base64(
        self, media_type: str, data: str
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        if media_type.lower() == "video/jpeg":
            load_frame = partial(
                self.image_io.load_base64,
                "image/jpeg",
            )

            return np.stack(
                [np.asarray(load_frame(frame_data)) for frame_data in data.split(",")]
            ), {}

        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, dict[str, Any]]:
        with filepath.open("rb") as f:
            data = f.read()

        return self.load_bytes(data)

    def encode_base64(
        self,
        media: npt.NDArray,
        *,
        video_format: str = "JPEG",
    ) -> str:
        video = media

        if video_format == "JPEG":
            encode_frame = partial(
                self.image_io.encode_base64,
                image_format=video_format,
            )

            return ",".join(encode_frame(Image.fromarray(frame)) for frame in video)

        msg = "Only JPEG format is supported for now."
        raise NotImplementedError(msg)
