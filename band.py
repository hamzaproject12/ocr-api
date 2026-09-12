"""Locating the machine-readable zone inside a page image.

OCR'ing a whole page never worked. The MRZ is a strip a few millimetres tall on
a passport that itself covers a fraction of an A4 scan, so Tesseract spends its
guesses on the photograph, the French and Arabic captions and the guilloche
background, and returns a line of plausible-looking rubbish that is the right
length and alphabet to pass for an MRZ.

The zone is found geometrically instead, before a single letter is read. MRZ
lines are set in OCR-B, which is monospaced: a line is a long run of similarly
sized glyphs at a constant pitch, and nothing else on a passport page looks like
that. Grouping character-sized connected components into rows and keeping the
rows whose glyph spacing is regular isolates the zone; what comes back is a
small, straightened, upscaled crop that Tesseract reads accurately.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

# A line holds 30 (TD1), 36 (TD2) or 44 (TD3) characters. The threshold sits
# below 30 because the thresholder loses the occasional faint glyph.
MIN_CHARS = 25
# Share of the gaps between neighbouring glyphs that must match the row's own
# pitch. Counting agreeing gaps rather than averaging their spread is what makes
# this survive a real scan: a speck of the background pattern, or a worn glyph
# broken in two, inserts a stray gap that would wreck a standard deviation but
# leaves the majority untouched. Proportional text agrees on nothing.
MIN_PITCH_AGREEMENT = 0.65
PITCH_TOLERANCE = 0.25
# Glyph height as a fraction of the page's short side, spanning a passport that
# fills the frame and one occupying a corner of an A4 sheet.
MIN_CHAR_FRACTION = 0.004
MAX_CHAR_FRACTION = 0.060
# Height at which Tesseract reads OCR-B most reliably.
TARGET_CHAR_HEIGHT = 40
# Rows this far apart, measured in glyph heights, belong to the same zone.
MAX_ROW_GAP = 2.5
MAX_BANDS = 4
# How far off square a page may be laid on the scanner and still have its rows
# grouped correctly, and how finely that range is searched. Half a degree is
# already under a glyph's height of drift across a full MRZ line.
MAX_SKEW = 5.0
SKEW_STEP = 0.25
# Glyphs are measured on a copy reduced to this short side. Detection is then
# insensitive to the resolution it is handed: a 600 dpi scan and a phone photo
# arrive at the thresholder with glyphs the same size, and the window below
# stays wider than a character instead of eating into one.
DETECT_SHORT_SIDE = 1600


def _flattened(gray: np.ndarray, window: int) -> np.ndarray:
    """Divide an image by its own background.

    Closing with a window wider than a glyph erases the text and leaves an
    estimate of the page behind it. Dividing by that estimate puts every glyph on
    the same white and, just as usefully, separates ink by how dark it started:
    the guilloche a passport is printed over survives as pale grey while the MRZ
    stays near black, so the wave lines stop bridging one character to the next.
    """
    background = cv2.morphologyEx(
        gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (window, window))
    )
    return cv2.divide(gray, background, scale=255)


def _page_window(gray: np.ndarray) -> int:
    return max(15, round(0.02 * min(gray.shape)) | 1)


def _binary(gray: np.ndarray) -> np.ndarray:
    """Ink mask of the page. Adaptive, because a scan lit unevenly or printed
    over a guilloche has no single global threshold.

    Deliberately not flattened first. Dividing the page by its background finds
    a few more zones on faded scans, but costs more than it wins: it thins the
    strokes of an already clean scan until glyphs break into pieces the row
    grouping can no longer see. Flattening earns its place on the crop, once the
    zone is known and its scale with it.
    """
    window = _page_window(gray)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, window, 12
    )


def _char_boxes(gray: np.ndarray) -> list[tuple]:
    """Connected components whose size and density make them plausible glyphs."""
    number, _, stats, centroids = cv2.connectedComponentsWithStats(_binary(gray), 8)
    short = min(gray.shape)
    smallest, largest = MIN_CHAR_FRACTION * short, MAX_CHAR_FRACTION * short
    boxes = []
    for index in range(1, number):
        x, y, width, height, area = stats[index]
        if not smallest <= height <= largest:
            continue
        # '<' is the widest glyph and 'I' the narrowest; anything outside this
        # is a rule, a speck of the background pattern or part of the photo.
        if not 0.10 <= width / height <= 1.6:
            continue
        if not 0.12 <= area / (width * height) <= 0.95:
            continue
        boxes.append((x, y, width, height, centroids[index][0], centroids[index][1]))
    return boxes


def skew(boxes: list[tuple]) -> float:
    """The angle, in degrees, at which the page's glyphs stack into sharpest rows.

    A sheet laid down by hand on a flatbed lands a degree or two off square, and
    over the width of an MRZ line that is more vertical drift than a row is
    allowed to have - the line breaks into pieces, each too short to be taken for
    a machine-readable zone. Rather than accept a looser row, which would let
    ordinary text chain together, find the tilt first.

    Projecting every glyph's centre onto a rotated vertical axis and squaring the
    resulting histogram is the standard measure: it peaks when the projection
    concentrates in a few tight bands, which is exactly when the angle is right.
    """
    if len(boxes) < MIN_CHARS:
        return 0.0
    xs = np.array([box[4] for box in boxes], dtype=float)
    ys = np.array([box[5] for box in boxes], dtype=float)
    xs -= xs.mean()
    ys -= ys.mean()
    bins = max(20, round((ys.max() - ys.min()) / max(np.median([b[3] for b in boxes]) / 3, 1)))
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-MAX_SKEW, MAX_SKEW + 1e-9, SKEW_STEP):
        radians = np.radians(angle)
        profile, _ = np.histogram(xs * np.sin(radians) + ys * np.cos(radians), bins=bins)
        score = float(np.square(profile.astype(float)).sum())
        if score > best_score:
            best_score, best_angle = score, float(angle)
    return best_angle


def _rows(boxes: list[tuple], angle: float = 0.0) -> list[list[tuple]]:
    """Group glyphs into rows by their vertical centre, along the page's own tilt.

    Boxes keep their true coordinates throughout; only the value they are sorted
    and cut on is measured along the tilted axis, so the crop and the deskew
    further down still work from where the glyphs actually are.
    """
    if not boxes:
        return []
    radians = np.radians(angle)
    sin_a, cos_a = np.sin(radians), np.cos(radians)

    def level(box: tuple) -> float:
        return box[4] * sin_a + box[5] * cos_a

    boxes = sorted(boxes, key=level)
    rows, current = [], [boxes[0]]
    # Measured against the row's first glyph rather than its last, so a page of
    # text that steps gently down cannot chain itself into one enormous row, and
    # against the glyphs' own height, so captions and headings each find their
    # own line.
    for box in boxes[1:]:
        tolerance = 0.6 * max(float(np.median([b[3] for b in current])), box[3])
        if level(box) - level(current[0]) <= tolerance:
            current.append(box)
        else:
            rows.append(current)
            current = [box]
    rows.append(current)
    return rows


def _agreement(row: list[tuple]) -> float:
    """Share of the row's glyph gaps that match its own pitch."""
    gaps = np.diff(np.sort(np.array([box[4] for box in row])))
    if len(gaps) == 0:
        return 0.0
    pitch = float(np.median(gaps))
    if pitch <= 0:
        return 0.0
    return float(np.mean(np.abs(gaps - pitch) <= PITCH_TOLERANCE * pitch))


def _is_regular(row: list[tuple]) -> bool:
    """True when the row's glyphs are monospaced, which only the MRZ is."""
    return len(row) >= MIN_CHARS and _agreement(row) >= MIN_PITCH_AGREEMENT


def _extent(row: list[tuple]) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in row),
        min(box[1] for box in row),
        max(box[0] + box[2] for box in row),
        max(box[1] + box[3] for box in row),
    )


def _centre(row: list[tuple]) -> float:
    """The row's vertical centre. Taken as a median rather than from whichever
    glyph happens to sort first, which on a tilted line is an edge, not a centre."""
    return float(np.median([box[5] for box in row]))


def _adjacent(first: list[tuple], second: list[tuple]) -> bool:
    """True when two rows sit one under the other and share a left and right edge,
    the way the lines of one machine-readable zone do."""
    height = float(np.median([box[3] for box in first]))
    if abs(_centre(second) - _centre(first)) > MAX_ROW_GAP * height:
        return False
    left, _, right, _ = _extent(first)
    other_left, _, other_right, _ = _extent(second)
    overlap = min(right, other_right) - max(left, other_left)
    return overlap > 0.5 * min(right - left, other_right - other_left)


def _group(rows: list[list[tuple]]) -> list[list[list[tuple]]]:
    """Collect rows into zones of 2 (TD2/TD3) or 3 (TD1) lines.

    Only a convincingly monospaced row may start a zone, but once one has, the
    rows stacked on it join whatever their own pitch looks like. A zone's second
    line is the one carrying the dates and the document number, where a stray
    mark is likeliest to spoil the measurement - and the line above has already
    established that this is the machine-readable zone.
    """
    rows = sorted(rows, key=_centre)
    zones: list[list[int]] = []
    for index, row in enumerate(rows):
        if not _is_regular(row):
            continue
        if zones and _adjacent(rows[zones[-1][-1]], row):
            zones[-1].append(index)
        else:
            zones.append([index])

    grouped = []
    for zone in zones:
        low, high = min(zone), max(zone)
        while low > 0 and high - low < 2 and _adjacent(rows[low], rows[low - 1]):
            low -= 1
        while high + 1 < len(rows) and high - low < 2 and _adjacent(rows[high], rows[high + 1]):
            high += 1
        grouped.append(rows[low:high + 1])
    return grouped


def _tilt(zone: list[list[tuple]]) -> float:
    """Median baseline tilt of the zone, in degrees."""
    slopes = []
    for row in zone:
        if len(row) < 2:
            continue
        xs = np.array([box[4] for box in row])
        ys = np.array([box[5] for box in row])
        slopes.append(np.polyfit(xs, ys, 1)[0])
    return float(np.degrees(np.arctan(np.median(slopes)))) if slopes else 0.0


def _crop(gray: np.ndarray, zone: list[list[tuple]], factor: float) -> Image.Image:
    """Cut the zone out, straighten it and scale it to Tesseract's best reading size.

    The zone was measured on the reduced copy; `factor` carries its coordinates
    back onto the full-resolution page, which is what actually gets read.
    """
    glyphs = [box for row in zone for box in row]
    height = float(np.median([box[3] for box in glyphs])) * factor
    left = min(_extent(row)[0] for row in zone) * factor
    top = min(_extent(row)[1] for row in zone) * factor
    right = max(_extent(row)[2] for row in zone) * factor
    bottom = max(_extent(row)[3] for row in zone) * factor
    # Generous vertically: if a line of the zone escaped detection entirely it is
    # still inside the crop, and Tesseract reads it along with the rest.
    pad_x, pad_y = round(height * 1.2), round(height * 1.6)
    patch = gray[
        max(0, round(top - pad_y)) : round(bottom + pad_y),
        max(0, round(left - pad_x)) : round(right + pad_x),
    ]
    if patch.size == 0:
        patch = gray

    angle = _tilt(zone)
    if abs(angle) > 0.3:
        centre = (patch.shape[1] / 2, patch.shape[0] / 2)
        matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
        patch = cv2.warpAffine(
            patch, matrix, (patch.shape[1], patch.shape[0]),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )

    scale = TARGET_CHAR_HEIGHT / max(height, 1.0)
    if scale > 1.0:
        patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(patch)


def _score(zone: list[list[tuple]]) -> tuple:
    """Rank zones: more rows first, then more glyphs, then the steadier pitch."""
    glyphs = sum(len(row) for row in zone)
    return (len(zone) >= 2, glyphs, max(_agreement(row) for row in zone))


def flatten(image: Image.Image) -> Image.Image:
    """Flatten a located zone against its own background.

    The same division as during detection, now at the crop's own scale. It is
    what stops Tesseract dropping the characters at the dim end of a line - the
    shadow across a page held open for a phone camera, or the darkening towards
    the gutter of a book scanner.
    """
    gray = np.array(image.convert("L"))
    return Image.fromarray(_flattened(gray, round(TARGET_CHAR_HEIGHT * 0.8) | 1))


def binarise(image: Image.Image) -> Image.Image:
    """Reduce a flattened zone to black on white, for the glyphs that survive the
    threshold better than they survive the grey."""
    _, black_on_white = cv2.threshold(
        np.array(flatten(image)), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    return Image.fromarray(black_on_white)


def candidates(image: Image.Image):
    """Yield MRZ crops, likeliest first, over every page orientation.

    A zone lying sideways on the page is found by looking at the page turned a
    quarter turn; a page that is upside down holds the same zone geometry as an
    upright one, so each crop is also offered rotated by half a turn.
    """
    found = []
    for quarter_turn in (False, True):
        page = image.transpose(Image.ROTATE_90) if quarter_turn else image
        gray = np.array(page)
        factor = max(1.0, min(gray.shape) / DETECT_SHORT_SIDE)
        reduced = (
            gray if factor == 1.0
            else cv2.resize(gray, (round(gray.shape[1] / factor), round(gray.shape[0] / factor)),
                            interpolation=cv2.INTER_AREA)
        )
        boxes = _char_boxes(reduced)
        for zone in _group(_rows(boxes, skew(boxes))):
            found.append((_score(zone), gray, zone, factor))
    found.sort(key=lambda item: item[0], reverse=True)

    for _, gray, zone, factor in found[:MAX_BANDS]:
        crop = _crop(gray, zone, factor)
        yield crop
        yield crop.transpose(Image.ROTATE_180)
