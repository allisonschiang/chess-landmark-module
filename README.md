# chess-landmark

Viam vision-service module `allisonorg:chess-landmark:detector`. It bundles a
synthesized computer-vision detector (`src/models/synth_detector.py`) and exposes it as a
vision service that returns Detections (a box of `box_size` px around each landmark) and
Classifications.

_Built by the landmark builder; `src/models/synth_detector.py`, `src/models/labels.txt`,
and `metadata.json` are regenerated each build. The rest is the module scaffold._

## Configure
Vision service attributes:
- `camera_name` (string, optional) — enables `*_from_camera` calls.
- `box_size` (int, optional, default 40) — side length in px of each detection box.

## Build & deploy
Cloud build runs on a pushed semver tag via `.github/workflows/deploy.yml`
(`viamrobotics/build-action`). Set repo secrets `viam_key_id` and `viam_key_value`.
Locally: `viam module build local` then `viam module build start --version=<v>`.
