#!/usr/bin/env python3
"""Test the detector against a live camera on a Viam machine.

Usage:
    python test_detector.py --address <robot-address> --api-key-id <id> --api-key <key> --camera <name>

Or set environment variables:
    export VIAM_ADDRESS=robot-main.xxxxx.viam.cloud
    export VIAM_API_KEY_ID=...
    export VIAM_API_KEY=...
    python test_detector.py --camera cam
"""

import argparse
import asyncio
import os
import sys

import cv2
import numpy as np
from viam.robot.client import RobotClient
from viam.components.camera import Camera

# Add src/ so we can import the detector directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from models.synth_detector import detect

# Distinct colours assigned round-robin to whatever labels the detector returns.
_PALETTE = [
    (0, 0, 255),      # red
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 165, 255),    # orange
    (128, 0, 128),    # purple
]


def _color_for(label: str, seen: dict) -> tuple:
    if label not in seen:
        seen[label] = _PALETTE[len(seen) % len(_PALETTE)]
    return seen[label]


async def run(args):
    address = args.address or os.environ.get("VIAM_ADDRESS", "")
    api_key_id = args.api_key_id or os.environ.get("VIAM_API_KEY_ID", "")
    api_key = args.api_key or os.environ.get("VIAM_API_KEY", "")

    if not address or not api_key_id or not api_key:
        print("Error: machine address, api-key-id, and api-key are required.")
        print("Pass them as flags or set VIAM_ADDRESS, VIAM_API_KEY_ID, VIAM_API_KEY.")
        sys.exit(1)

    opts = RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id)
    print(f"Connecting to {address} ...")
    robot = await RobotClient.at_address(address, opts)
    print("Connected.")

    try:
        camera = Camera.from_robot(robot, args.camera)
        count = 0
        label_colors: dict = {}

        while True:
            img = await camera.get_image()

            # Convert PIL Image to BGR numpy array for cv2 / detector.
            rgb = np.array(img)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            results = detect(bgr)
            if args.labels:
                results = {k: v for k, v in results.items() if k in args.labels}
            count += 1

            # Print detections.
            print(f"\n--- frame {count} ({bgr.shape[1]}x{bgr.shape[0]}) ---")
            for label, (x, y) in sorted(results.items()):
                print(f"  {label}: ({x:.1f}, {y:.1f})")

            # Draw on image.
            display = bgr.copy()
            for label, (x, y) in results.items():
                color = _color_for(label, label_colors)
                cx, cy = int(round(x)), int(round(y))
                cv2.circle(display, (cx, cy), 8, color, -1)
                cv2.circle(display, (cx, cy), 10, (255, 255, 255), 2)
                cv2.putText(display, label, (cx + 14, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("detector test", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = f"snapshot_{count}.jpg"
                cv2.imwrite(fname, display)
                print(f"  saved {fname}")

            if args.once:
                cv2.waitKey(0)
                break
    finally:
        await robot.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Test the detector against a live Viam camera."
    )
    parser.add_argument("--address", default="",
                        help="Robot address (or VIAM_ADDRESS env var)")
    parser.add_argument("--api-key-id", default="",
                        help="API key ID (or VIAM_API_KEY_ID env var)")
    parser.add_argument("--api-key", default="",
                        help="API key (or VIAM_API_KEY env var)")
    parser.add_argument("--camera", required=True,
                        help="Name of the camera component")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Only show detections for these labels (default: all)")
    parser.add_argument("--once", action="store_true",
                        help="Capture one frame, display, wait for keypress, then exit")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
