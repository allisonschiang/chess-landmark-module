"""Vision service wrapping the synthesized detector (bundled as synth_detector.py).

The detector exposes `detect(image_bgr) -> {label: (x, y)}`. Each point becomes a
square Detection (side = the `box_size` config attribute, default 40px) and a
Classification. Configure with attributes: `camera_name` (optional, enables the
*_from_camera methods) and `box_size` (optional).
"""
from typing import ClassVar, List, Mapping, Optional, Sequence, Tuple, cast

import cv2
import numpy as np
from typing_extensions import Self
from viam.components.camera import Camera
from viam.media.video import ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import CaptureAllResult, Vision
from viam.utils import ValueTypes, struct_to_dict

from models.synth_detector import detect

DEFAULT_BOX_SIZE = 40


class Detector(Vision, EasyResource):
    MODEL: ClassVar[Model] = Model(
        ModelFamily("allisonorg", "chess-landmark"), "detector"
    )

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        return super().new(config, dependencies)

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        camera_name = attrs.get("camera_name")
        optional = [camera_name] if isinstance(camera_name, str) and camera_name else []
        return [], optional

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = struct_to_dict(config.attributes)
        self.box_size = int(attrs.get("box_size") or DEFAULT_BOX_SIZE)
        self.camera_name = str(attrs.get("camera_name") or "")
        self.dependencies = dependencies

    # --- detector plumbing -------------------------------------------------
    @staticmethod
    def _to_bgr(image: ViamImage) -> np.ndarray:
        bgr = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("could not decode image bytes")
        return bgr

    def _points(self, image: ViamImage) -> List[Tuple[str, float, float]]:
        result = detect(self._to_bgr(image)) or {}
        points: List[Tuple[str, float, float]] = []
        for label, value in result.items():
            many = (isinstance(value, (list, tuple)) and value
                    and isinstance(value[0], (list, tuple)))
            for p in (value if many else [value]):
                points.append((label, float(p[0]), float(p[1])))
        return points

    def _detections(self, image: ViamImage) -> List[Detection]:
        half = self.box_size / 2.0
        return [
            Detection(
                x_min=int(x - half), y_min=int(y - half),
                x_max=int(x + half), y_max=int(y + half),
                confidence=1.0, class_name=label,
            )
            for label, x, y in self._points(image)
        ]

    def _classifications(
        self, image: ViamImage, count: Optional[int] = None
    ) -> List[Classification]:
        out = [Classification(class_name=label, confidence=1.0)
               for label, _, _ in self._points(image)]
        return out[:count] if count else out

    async def _camera_image(self, camera_name: str) -> ViamImage:
        name = camera_name or self.camera_name
        if not name:
            raise ValueError("no camera configured; set the `camera_name` attribute")
        camera = cast(Camera, self.dependencies[Camera.get_resource_name(name)])
        images, _ = await camera.get_images()
        if not images:
            raise ValueError(f"camera {name} returned no images")
        return images[0]

    # --- Vision API --------------------------------------------------------
    async def get_detections(
        self, image: ViamImage, *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        return self._detections(image)

    async def get_detections_from_camera(
        self, camera_name: str, *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        return self._detections(await self._camera_image(camera_name))

    async def get_classifications(
        self, image: ViamImage, count: int, *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        return self._classifications(image, count)

    async def get_classifications_from_camera(
        self, camera_name: str, count: int, *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        return self._classifications(await self._camera_image(camera_name), count)

    async def capture_all_from_camera(
        self, camera_name: str,
        return_image: bool = False, return_classifications: bool = False,
        return_detections: bool = False, return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        result = CaptureAllResult()
        if not (return_image or return_classifications or return_detections):
            return result
        image = await self._camera_image(camera_name)
        if return_image:
            result.image = image
        if return_detections:
            result.detections = self._detections(image)
        if return_classifications:
            result.classifications = self._classifications(image)
        return result

    async def get_object_point_clouds(
        self, camera_name: str, *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        raise NotImplementedError()

    async def get_properties(
        self, *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Vision.Properties:
        return Vision.Properties(
            classifications_supported=True,
            detections_supported=True,
            object_point_clouds_supported=False,
        )

    async def do_command(
        self, command: Mapping[str, ValueTypes], *,
        timeout: Optional[float] = None, **kwargs,
    ) -> Mapping[str, ValueTypes]:
        raise NotImplementedError()
