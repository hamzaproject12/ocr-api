"""Parsing of ICAO 9303 machine-readable zones (MRZ).

Supports the three standard layouts:
  * TD1 - identity cards, 3 lines of 30 characters (Moroccan CNIE, EU ID cards)
  * TD2 - travel documents, 2 lines of 36 characters
  * TD3 - passports, 2 lines of 44 characters

Every field the standard protects with a check digit is validated, and the check
digit is also used to arbitrate between competing OCR readings: Tesseract
reliably confuses the filler '<' with 'K', and digits with their look-alike
letters, so each field is retried with those substitutions until its check digit
matches. That is what replaces the previous blanket 'K' -> '<' replacement,
which also destroyed the legitimate K's in names and document numbers.
"""

from __future__ import annotations

import re
from datetime import date

MRZ_CHARSET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")

# Unicode shapes Tesseract emits in place of the filler chevron.
_LOOKALIKES = {"«": "<<", "»": "<<", "‹": "<", "›": "<",
               "＜": "<", "〈": "<", "≪": "<<"}

# Look-alike fixes applied inside a field whose content type is known.
_TO_DIGIT = str.maketrans("OQDIL|ZSBGT", "00011125867")
_TO_ALPHA = str.maketrans("012358", "OIZBSB")

_WEIGHTS = (7, 3, 1)

# (line length, number of lines, format), in the order we try them.
_LAYOUTS = ((44, 2, "TD3"), (36, 2, "TD2"), (30, 3, "TD1"))


def _value(char: str) -> int | None:
    if char.isdigit():
        return int(char)
    if char == "<":
        return 0
    if "A" <= char <= "Z":
        return ord(char) - 55
    return None


def check_digit(field: str) -> str | None:
    """The ICAO 9303 check digit of `field`, or None if it holds an invalid character."""
    total = 0
    for index, char in enumerate(field):
        value = _value(char)
        if value is None:
            return None
        total += value * _WEIGHTS[index % 3]
    return str(total % 10)


def _variants(field: str, kind: str) -> list[str]:
    """Plausible readings of a field, most conservative first."""
    readings: list[str] = []

    def add(candidate: str) -> None:
        if candidate and candidate not in readings:
            readings.append(candidate)

    if kind == "num":
        add(field.translate(_TO_DIGIT))
    elif kind == "alpha":
        add(field.translate(_TO_ALPHA))
    add(field)
    # Trailing K's are padding that came back as a letter.
    add(re.sub(r"K+$", lambda m: "<" * len(m.group()), field))
    # A run of two or more K's is a filler sequence, never part of a real value.
    add(re.sub(r"K{2,}", lambda m: "<" * len(m.group()), field))
    add(field.replace("K", "<"))
    return readings


def _read(line: str, start: int, end: int, check_at: int | None = None, kind: str = "raw"):
    """Return (value, check_ok) for one field, repairing OCR errors when a check digit is available."""
    field = line[start:end]
    readings = _variants(field, kind)
    if check_at is None:
        return readings[0], None

    expected = line[check_at]
    accepted = {expected, expected.translate(_TO_DIGIT)}
    for candidate in readings:
        # An unused optional field carries either '0' or a filler as its check digit.
        if set(candidate) == {"<"} and expected in "<0":
            return candidate, True
        if check_digit(candidate) in accepted:
            return candidate, True
    return readings[0], False


def _clean(field: str) -> str:
    return field.replace("<", "").strip()


def _sex(char: str) -> str:
    return char if char in ("M", "F") else ""


def _split_names(field: str) -> tuple[str, str]:
    """Split an MRZ name field into (surname, given names)."""
    # Drop the padding first. Tesseract renders the trailing run of chevrons as
    # whatever letter it takes them for - K, S, C - so strip any run of three or
    # more identical characters at the end, repeatedly, since the run and the
    # chevrons it failed to read are usually interleaved. A single trailing
    # letter is part of the name (MALIK).
    previous = None
    while previous != field:
        previous = field
        field = re.sub(r"(.)\1{2,}$", "", field.rstrip("<"))
    # Only then guess at KK-as-separator, and only if no real one survived.
    if "<<" not in field and re.search(r"K{2,}", field):
        field = re.sub(r"K{2,}", lambda m: "<" * len(m.group()), field)

    surname, _, given = field.partition("<<")

    def tidy(part: str) -> str:
        return re.sub(r"\s+", " ", part.replace("<", " ")).strip()

    return tidy(surname), tidy(given)


def to_iso(yymmdd: str, *, past: bool) -> str | None:
    """Turn a YYMMDD MRZ date into an ISO date. `past` picks the century window."""
    if not re.fullmatch(r"\d{6}", yymmdd):
        return None
    year, month, day = int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    # A birth date cannot be in the future; an expiry date is written 20xx.
    year += 2000 if not past or 2000 + year <= date.today().year else 1900
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _result(fmt, lines, *, surname, given, number, nationality, birth, sex, expiry, personal, checks):
    return {
        "document_format": fmt,
        "mrz": lines,
        "checks": checks,
        "fields": {
            "first_name": given,
            "last_name": surname,
            "passport": _clean(number),
            "nationality": _clean(nationality),
            "date_of_birth": birth,
            "sex": sex,
            "date_of_expiry": expiry,
            "personal_id_number": _clean(personal),
        },
    }


def _parse_td3(lines: list[str]) -> dict:
    line1, line2 = lines
    surname, given = _split_names(line1[5:])
    number, number_ok = _read(line2, 0, 9, 9)
    nationality, _ = _read(line2, 10, 13, kind="alpha")
    birth, birth_ok = _read(line2, 13, 19, 19, "num")
    expiry, expiry_ok = _read(line2, 21, 27, 27, "num")
    personal, personal_ok = _read(line2, 28, 42, 42)
    return _result(
        "TD3", lines,
        surname=surname, given=given, number=number, nationality=nationality,
        birth=birth, sex=_sex(line2[20]), expiry=expiry, personal=personal,
        checks={"passport": number_ok, "date_of_birth": birth_ok,
                "date_of_expiry": expiry_ok, "personal_id_number": personal_ok},
    )


def _parse_td2(lines: list[str]) -> dict:
    line1, line2 = lines
    surname, given = _split_names(line1[5:])
    number, number_ok = _read(line2, 0, 9, 9)
    nationality, _ = _read(line2, 10, 13, kind="alpha")
    birth, birth_ok = _read(line2, 13, 19, 19, "num")
    expiry, expiry_ok = _read(line2, 21, 27, 27, "num")
    return _result(
        "TD2", lines,
        surname=surname, given=given, number=number, nationality=nationality,
        birth=birth, sex=_sex(line2[20]), expiry=expiry, personal=line2[28:35],
        checks={"passport": number_ok, "date_of_birth": birth_ok, "date_of_expiry": expiry_ok},
    )


def _parse_td1(lines: list[str]) -> dict:
    line1, line2, line3 = lines
    number, number_ok = _read(line1, 5, 14, 14)
    birth, birth_ok = _read(line2, 0, 6, 6, "num")
    expiry, expiry_ok = _read(line2, 8, 14, 14, "num")
    nationality, _ = _read(line2, 15, 18, kind="alpha")
    surname, given = _split_names(line3)
    # The national identity number (the Moroccan CNIE included) sits in one of
    # the two optional-data areas, depending on the issuing country.
    personal = _clean(line1[15:30]) or _clean(line2[18:29])
    return _result(
        "TD1", lines,
        surname=surname, given=given, number=number, nationality=nationality,
        birth=birth, sex=_sex(line2[7]), expiry=expiry, personal=personal,
        checks={"passport": number_ok, "date_of_birth": birth_ok, "date_of_expiry": expiry_ok},
    )


_PARSERS = {"TD1": _parse_td1, "TD2": _parse_td2, "TD3": _parse_td3}


def _compact(raw: str) -> str:
    text = raw.upper()
    for shape, replacement in _LOOKALIKES.items():
        text = text.replace(shape, replacement)
    return re.sub(r"\s+", "", text)


def _mrz_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(char in MRZ_CHARSET for char in text) / len(text)


def _fit(line: str, size: int) -> str:
    return line[:size] if len(line) > size else line.ljust(size, "<")


def _candidate_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        compact = _compact(raw)
        if len(compact) < 26 or _mrz_ratio(compact) < 0.8:
            continue
        stripped = "".join(char for char in compact if char in MRZ_CHARSET)
        # Tesseract sometimes returns the whole zone glued into a single line.
        for size, count, _ in _LAYOUTS:
            if abs(len(stripped) - size * count) <= 1:
                lines.extend(stripped[i * size:(i + 1) * size] for i in range(count))
                break
        else:
            lines.append(stripped)
    return lines


def _candidate_groups(text: str):
    """Yield (format, lines) for every group of lines that could be an MRZ."""
    lines = _candidate_lines(text)
    for size, count, fmt in _LAYOUTS:
        for start in range(len(lines) - count + 1):
            group = lines[start:start + count]
            if any(abs(len(line) - size) > 2 for line in group):
                continue
            # The first character is the document code: P for passports,
            # I/A/C for identity cards and other travel documents.
            if group[0][0] not in ("PIAC" if count == 2 else "IAC"):
                continue
            yield fmt, [_fit(line, size) for line in group]


def parse(text: str) -> dict | None:
    """Extract the best machine-readable zone from `text`, or None if there is none."""
    best = None
    for fmt, lines in _candidate_groups(text):
        result = _PARSERS[fmt](lines)
        checks = result["checks"]
        score = sum(1 for ok in checks.values() if ok)
        if best is None or score > best[0]:
            best = (score, result)
        if score == len(checks):
            break
    return best[1] if best else None
