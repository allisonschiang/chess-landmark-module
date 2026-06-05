"""Chess piece occupancy + color detector.

Given a top-down chess board image (BGR uint8 HxWx3), returns a dict with
'white' and 'black' label keys mapped to (x, y) point or list of points at
each detected piece center.

Algorithm:
1. Estimate board rotation from gradient-orientation histogram in central band
   (independent of grid fit -- prevents wrong-tilt lock-in).
2. De-rotate the image.
3. |Sobel| projection profiles along the central row/col band give vertical /
   horizontal grid-line strengths.
4. Fit a 9-line arithmetic progression (a + k*T, k=0..8) to each profile -> outer
   grid lines and cell size in rotated coords.
5. Rotate cell-center coordinates back to original image space.
6. Extend the grid +/-2 cells in every direction to cover off-board captures.
7. Build HSV color masks for WHITE and BLACK pieces.
8. For each extended cell, compute fraction of patch in white_mask and
   black_mask. Classify as white / black / empty.
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------- Grid detection ----------------

def _estimate_angle(gray: np.ndarray) -> float:
    """Magnitude-weighted gradient-orientation histogram (mod 90) over the
    central image band -> degrees to rotate to make grid lines axis-aligned.
    Returns angle in (-45, 45]."""
    h, w = gray.shape
    cb = gray[int(0.34 * h):int(0.66 * h), int(0.28 * w):int(0.72 * w)]
    cb = cv2.GaussianBlur(cb, (5, 5), 1.0)
    gx = cv2.Sobel(cb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(cb, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx) * 180.0 / np.pi
    ang_mod = np.mod(ang, 90.0)
    bins = 360
    hist, edges = np.histogram(ang_mod.ravel(), bins=bins, range=(0, 90),
                               weights=mag.ravel())
    hk = cv2.getGaussianKernel(15, 3).flatten()
    hist_sm = np.convolve(hist, hk, mode='same')
    peak = int(np.argmax(hist_sm))
    lo = max(0, peak - 7)
    hi = min(bins, peak + 8)
    centers = 0.5 * (edges[:-1] + edges[1:])
    angle = float((centers[lo:hi] * hist_sm[lo:hi]).sum() /
                  (hist_sm[lo:hi].sum() + 1e-9))
    if angle > 45.0:
        angle -= 90.0
    return angle


def _ap_score(profile: np.ndarray, a_arr: np.ndarray, T: float, n: int) -> np.ndarray:
    L = len(profile)
    score = np.zeros(len(a_arr), dtype=np.float32)
    for k in range(n):
        pos = a_arr + k * T
        ipos = pos.astype(np.int32)
        frac = pos - ipos
        ipos = np.clip(ipos, 0, L - 2)
        v = profile[ipos] * (1.0 - frac) + profile[ipos + 1] * frac
        score += v
    return score


def _fit_ap(profile: np.ndarray, T_min: float, T_max: float, n: int = 9):
    """Return (a, T) maximizing sum of profile at a + k*T for k in [0, n)."""
    L = len(profile)
    prof = profile.astype(np.float32)
    best_score = -1e18
    best_a = 0.0
    best_T = 0.0
    Ts = np.arange(T_min, T_max + 1, 1.0, dtype=np.float32)
    for T in Ts:
        if (n - 1) * T >= L:
            continue
        a_max = L - (n - 1) * T - 1
        a_arr = np.arange(0.0, a_max + 1.0, 1.0, dtype=np.float32)
        score = _ap_score(prof, a_arr, float(T), n)
        idx = int(np.argmax(score))
        if score[idx] > best_score:
            best_score = float(score[idx])
            best_a = float(a_arr[idx])
            best_T = float(T)
    T0, a0 = best_T, best_a
    for T in np.arange(T0 - 1.5, T0 + 1.5 + 1e-6, 0.2):
        a_lo = max(0.0, a0 - 3.0)
        a_hi = a0 + 3.0 + 1e-6
        a_arr = np.arange(a_lo, a_hi, 0.2, dtype=np.float32)
        if len(a_arr) == 0:
            continue
        score = _ap_score(prof, a_arr, float(T), n)
        idx = int(np.argmax(score))
        if score[idx] > best_score:
            best_score = float(score[idx])
            best_a = float(a_arr[idx])
            best_T = float(T)
    return best_a, best_T


def _detect_grid(img_bgr: np.ndarray):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    angle = _estimate_angle(gray)

    cx, cy = w / 2.0, h / 2.0
    M_fwd = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(gray, M_fwd, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    rh, rw = rotated.shape
    band_rows = rotated[int(0.30 * rh):int(0.70 * rh), :]
    band_cols = rotated[:, int(0.30 * rw):int(0.70 * rw)]
    gx = cv2.Sobel(band_rows, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(band_cols, cv2.CV_32F, 0, 1, ksize=3)
    prof_x = np.abs(gx).sum(axis=0)
    prof_y = np.abs(gy).sum(axis=1)

    k = cv2.getGaussianKernel(7, 1.0).flatten()
    prof_x = np.convolve(prof_x, k, mode='same')
    prof_y = np.convolve(prof_y, k, mode='same')

    T_min, T_max = 55.0, 105.0
    ax, Tx = _fit_ap(prof_x, T_min, T_max, n=9)
    ay, Ty = _fit_ap(prof_y, T_min, T_max, n=9)

    M_inv = cv2.invertAffineTransform(M_fwd)

    def rot_to_img(x, y):
        X = M_inv[0, 0] * x + M_inv[0, 1] * y + M_inv[0, 2]
        Y = M_inv[1, 0] * x + M_inv[1, 1] * y + M_inv[1, 2]
        return X, Y

    origin = np.array(rot_to_img(ax, ay), dtype=np.float64)
    ex_end = np.array(rot_to_img(ax + Tx, ay), dtype=np.float64)
    ey_end = np.array(rot_to_img(ax, ay + Ty), dtype=np.float64)
    ex = ex_end - origin
    ey = ey_end - origin
    cell = 0.5 * (np.linalg.norm(ex) + np.linalg.norm(ey))

    return {
        'origin': origin,
        'ex': ex,
        'ey': ey,
        'cell': cell,
        'angle': angle,
        'Tx': Tx,
        'Ty': Ty,
    }


# ---------------- Piece masks ----------------

def _color_masks(img_bgr: np.ndarray):
    """Return (white_mask, black_mask) uint8 0/255.

    Two distinct piece styles appear in the dataset:
    A. Warm pinkish-cream white pieces (V>=180, S 22..160) on green or dark
       brown boards. The S floor excludes cream board squares / frame (S<15).
    B. Pure white pieces (V>=225, S<=30) on a lighter brown board. Cream
       squares on that board have V<=215 so V>=225 separates them.

    Black pieces span warm-tinted darker (S~30-60, H~30-50) and blue-tinted
    (S~80-110, H~95-100). Single threshold V<=70 captures both. Dark green
    squares (S>120) are excluded.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    white = ((V >= 180) & (S >= 22) & (S <= 160)).astype(np.uint8) * 255

    black = ((V <= 70) & (S <= 125)).astype(np.uint8) * 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k3, iterations=1)
    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, k3, iterations=1)
    return white, black


def _filter_large_ccs(mask: np.ndarray, max_area: float) -> np.ndarray:
    """Drop any connected component whose area exceeds max_area pixels.
    A single chess piece on a coloured board is typically <= cell^2.
    The dark wood table off-board forms a huge CC and is removed this way."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[:, cv2.CC_STAT_AREA]
    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[1:] = areas[1:] <= max_area
    keep_pix = keep_mask[labels]
    return np.where(keep_pix, 255, 0).astype(np.uint8)


# ---------------- Per-cell classification ----------------

def _cc_at_point(labels, stats, x, y):
    H, W = labels.shape
    xi = int(round(x)); yi = int(round(y))
    if not (0 <= xi < W and 0 <= yi < H):
        return None
    lab = int(labels[yi, xi])
    if lab == 0:
        return None
    area = int(stats[lab, cv2.CC_STAT_AREA])
    return lab, area


def _classify_on_board(image_bgr, white_mask, black_mask, grid):
    """Classify cells INSIDE the 8x8 playing grid only.

    Two-stage:
    1. CC-at-pixel on colour masks (warm cream / pinkish white pieces;
       dark black pieces). Area + centroid validation.
    2. V-percentile + Sobel fallback for cells where colour mask is empty:
       e.g. pure-white pieces on the second-style board that have S<22 and
       therefore don't match the warm-cream white mask.
    """
    H, W = image_bgr.shape[:2]
    origin = grid['origin']
    ex = grid['ex']
    ey = grid['ey']
    cell = grid['cell']

    nw, wlabels, wstats, wcent = cv2.connectedComponentsWithStats(white_mask, 8)
    nb, blabels, bstats, bcent = cv2.connectedComponentsWithStats(black_mask, 8)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    Vall = hsv[..., 2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gxk = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gyk = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sob = np.sqrt(gxk * gxk + gyk * gyk)

    R = max(6, int(round(0.28 * cell)))
    max_centroid_offset = 0.50 * cell
    min_cc_area = max(120.0, 0.05 * cell * cell)
    max_cc_area = 3.0 * cell * cell

    cells = {}
    for j in range(8):
        for i in range(8):
            c = origin + (i + 0.5) * ex + (j + 0.5) * ey
            cx, cy = float(c[0]), float(c[1])

            w_score = 0.0
            b_score = 0.0
            r = _cc_at_point(wlabels, wstats, cx, cy)
            if r is not None:
                lab, area = r
                if min_cc_area <= area <= max_cc_area:
                    ccx, ccy = float(wcent[lab, 0]), float(wcent[lab, 1])
                    dist = ((ccx - cx) ** 2 + (ccy - cy) ** 2) ** 0.5
                    if dist <= max_centroid_offset:
                        w_score = float(area) / (cell * cell)
            r = _cc_at_point(blabels, bstats, cx, cy)
            if r is not None:
                lab, area = r
                if min_cc_area <= area <= max_cc_area:
                    ccx, ccy = float(bcent[lab, 0]), float(bcent[lab, 1])
                    dist = ((ccx - cx) ** 2 + (ccy - cy) ** 2) ** 0.5
                    if dist <= max_centroid_offset:
                        b_score = float(area) / (cell * cell)

            if w_score == 0.0 and b_score == 0.0:
                x0 = int(round(cx - R)); x1 = int(round(cx + R))
                y0 = int(round(cy - R)); y1 = int(round(cy + R))
                xa = max(0, x0); xb = min(W, x1)
                ya = max(0, y0); yb = min(H, y1)
                if xb - xa >= 6 and yb - ya >= 6:
                    patch_area = (xb - xa) * (yb - ya)
                    Vp = Vall[ya:yb, xa:xb]
                    v_hi = float(np.percentile(Vp, 90))
                    v_lo = float(np.percentile(Vp, 10))
                    contrast = v_hi - v_lo
                    sob_mean = float(sob[ya:yb, xa:xb].mean())
                    wm = white_mask[ya:yb, xa:xb]
                    bm = black_mask[ya:yb, xa:xb]
                    pw = float((wm > 0).sum()) / patch_area
                    pb = float((bm > 0).sum()) / patch_area

                    if pw >= 0.18 and pw >= pb:
                        w_score = pw + 0.5
                    elif pb >= 0.30 and pb >= pw:
                        b_score = pb + 0.5
                    if w_score == 0.0 and b_score == 0.0:
                        if v_hi >= 230 and contrast >= 25 and sob_mean >= 18:
                            w_score = 0.5 + (v_hi - 230) / 50.0
                        elif v_lo <= 55 and contrast >= 25 and sob_mean >= 18:
                            b_score = 0.5 + (60 - v_lo) / 50.0

            label = None
            if w_score > 0 and (b_score == 0 or w_score >= b_score):
                label = 'white'
            elif b_score > 0:
                label = 'black'

            if label is not None:
                cells[(i, j)] = (cx, cy, label, w_score, b_score)
    return cells


def _pure_white_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Mask for pure-white pieces (V>=225, S<=30). Cream playing squares and
    cream board frame are also caught by this but are large CCs that get
    filtered out at the caller."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1]; V = hsv[..., 2]
    m = ((V >= 225) & (S <= 30)).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3, iterations=1)
    return m


def _detect_captures(image_bgr, white_mask, black_mask, grid, pad=2):
    """Detect captured pieces OFF the 8x8 playing grid.

    Restricted to the FAR-LEFT (i=-2) and FAR-RIGHT (i=9) columns at playing
    rows j in [0, 7] -- where >80% of all observed captures sit.

    Per candidate cell, computes per-cell V-percentile and Sobel statistics
    (similar to on-board) but with STRICTER thresholds because the
    off-board background is wood (sometimes wood-grain matches the colour
    masks, which is too noisy for liberal rules).
    """
    H, W = image_bgr.shape[:2]
    origin = grid['origin']
    ex = grid['ex']
    ey = grid['ey']
    cell = grid['cell']

    pure_white = _pure_white_mask(image_bgr)
    pure_white = _filter_large_ccs(pure_white, 10.0 * cell * cell)
    white_comb = cv2.bitwise_or(white_mask, pure_white)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    Vall = hsv[..., 2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gxk = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gyk = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sob = np.sqrt(gxk * gxk + gyk * gyk)

    R = max(8, int(round(0.32 * cell)))
    max_centroid_offset = 0.30 * cell

    out = {}
    # Empirically captures appear in a narrow band:
    #   i = -2 -> j in [1, 3]   (84+23+6 captures)
    #   i =  9 -> j in [4, 6]   (49+24+3 captures)
    # Cells outside that band have heavy FP sources (frame letters,
    # corner Post-It notes, table edges) so we skip them entirely.
    candidate_cells = [(-2, 1), (-2, 2), (-2, 3),
                       (9, 4), (9, 5), (9, 6)]
    for ic, jc in candidate_cells:
        if True:
            c = origin + (ic + 0.5) * ex + (jc + 0.5) * ey
            cx, cy = float(c[0]), float(c[1])
            x0 = int(round(cx - R)); x1 = int(round(cx + R))
            y0 = int(round(cy - R)); y1 = int(round(cy + R))
            xa = max(0, x0); xb = min(W, x1)
            ya = max(0, y0); yb = min(H, y1)
            if xb - xa < 6 or yb - ya < 6:
                continue
            patch_area = (xb - xa) * (yb - ya)
            Vp = Vall[ya:yb, xa:xb]
            v_mean = float(Vp.mean())
            v_hi = float(np.percentile(Vp, 90))
            v_lo = float(np.percentile(Vp, 10))
            contrast = v_hi - v_lo
            sob_mean = float(sob[ya:yb, xa:xb].mean())
            wm = white_comb[ya:yb, xa:xb]
            bm = black_mask[ya:yb, xa:xb]
            pw = float((wm > 0).sum()) / patch_area
            pb = float((bm > 0).sum()) / patch_area

            def offset(mask_patch):
                ys, xs = np.nonzero(mask_patch)
                if len(ys) < 20:
                    return 1e9
                return ((xs.mean() + xa - cx) ** 2 +
                        (ys.mean() + ya - cy) ** 2) ** 0.5

            label = None
            # WHITE capture: very bright peak OR solid colour-mask coverage.
            if ((v_hi >= 235 and contrast >= 40 and sob_mean >= 22
                 and offset(wm) <= max_centroid_offset) or
                (pw >= 0.30 and sob_mean >= 20
                 and offset(wm) <= max_centroid_offset)):
                label = 'white'
            # BLACK capture: V_lo very low (clear dark piece) with non-trivial
            # Sobel and a colour-mask hint (catches piece-on-wood where the
            # piece itself doesn't form a clean CC after filter).
            if label is None:
                if (v_lo <= 35 and contrast >= 30 and sob_mean >= 22
                        and offset(bm) <= max_centroid_offset):
                    label = 'black'

            if label is not None:
                out[(ic, jc)] = (cx, cy, label, pw, pb)
    return out


def _classify_cells(image_bgr, white_mask, black_mask, grid, pad=2):
    on = _classify_on_board(image_bgr, white_mask, black_mask, grid)
    off = _detect_captures(image_bgr, white_mask, black_mask, grid, pad=pad)
    on.update(off)
    return on


# ---------------- Public API ----------------

def detect(image_bgr):
    H, W = image_bgr.shape[:2]
    grid = _detect_grid(image_bgr)
    cell = grid['cell']
    if not (40.0 < cell < 140.0):
        return {}

    white_mask, black_mask = _color_masks(image_bgr)
    # A single piece's mask blob is on the order of cell^2 in area. Drop any
    # connected component bigger than ~10 cells^2 (the wood table off-board,
    # a Post-It note, large shadows, etc.).
    max_cc = 10.0 * cell * cell
    white_mask = _filter_large_ccs(white_mask, max_cc)
    black_mask = _filter_large_ccs(black_mask, max_cc)
    cells = _classify_cells(image_bgr, white_mask, black_mask, grid, pad=2)

    white_pts = []
    black_pts = []
    for (i, j), (cx, cy, lab, w_score, b_score) in cells.items():
        if lab == 'white':
            white_pts.append((cx, cy))
        elif lab == 'black':
            black_pts.append((cx, cy))

    result = {}
    if white_pts:
        result['white'] = white_pts if len(white_pts) > 1 else white_pts[0]
    if black_pts:
        result['black'] = black_pts if len(black_pts) > 1 else black_pts[0]
    return result
