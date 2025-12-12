#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
r"""
Update the rustc manpage "-O" description to match `rustc --help`.

Usage (dry-run by default):
  ./src/tools/update-rustc-man-opt-level.py \
      --man-file src/doc/man/rustc.1

Apply changes (creates a timestamped backup):
  ./src/tools/update-rustc-man-opt-level.py \
      --man-file src/doc/man/rustc.1 --apply

Force a level instead of querying rustc:
  ./src/tools/update-rustc-man-opt-level.py \
      --man-file ... --expected-level 3 --apply
"""

from __future__ import annotations

import argparse
import datetime
import difflib
import shutil
import subprocess
import sys
import re
from pathlib import Path

DEFAULT_RUSTC = "rustc"

# ANSI color codes
_CLR = {
    "reset": "\033[0m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "bold": "\033[1m",
}


def colorize(line: str, color: str, enabled: bool) -> str:
    if not enabled or color not in _CLR:
        return line
    return f"{_CLR[color]}{line}{_CLR['reset']}"


def get_rustc_opt_level(rustc_cmd: str = DEFAULT_RUSTC) -> int:
    """Query `rustc --help` and parse the opt-level mapped to -O."""
    try:
        proc = subprocess.run(
            [rustc_cmd, "--help"], capture_output=True, text=True, check=True
        )
    except FileNotFoundError:
        raise RuntimeError(f"rustc not found at '{rustc_cmd}'")
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        raise RuntimeError(f"rustc --help failed: {stderr or e}") from e

    help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    # Fixed: removed unnecessary \\ escape before the hyphen
    m = re.search(r"-O[^\n]*opt(?:-)?level\s*=\s*(\d+)", help_text, flags=re.IGNORECASE)
    if not m:
        m2 = re.search(
            r"Equivalent to\s+-C\s+opt(?:-)?level\s*=\s*(\d+)",
            help_text,
            flags=re.IGNORECASE,
        )
        if not m2:
            raise RuntimeError(
                "Could not find '-O' opt-level mapping in rustc --help output"
            )
        return int(m2.group(1))
    return int(m.group(1))


def find_and_replace_manpage_content(
    content: str, new_level: int
) -> tuple[str, int, bool]:
    r"""
    Replace opt-level numbers in 'Equivalent to ... opt-level=N.'
    sentences tied to -O.

    Conservative heuristic:
      - Locate sentences starting with 'Equivalent to' up to the
        next period.
      - Ensure the sentence mentions opt-level (accepting escaped
        '\-').
      - Confirm a -O header appears within a lookback window before
        the sentence.

    Returns:
        tuple of (new_content, replacements_made, found_any_patterns)
    """
    replacements = 0
    found_patterns = False
    out_parts = []
    last_index = 0

    # More conservative limit of 200 chars instead of 800
    sentence_pattern = re.compile(
        r"Equivalent to([^\n\.]{0,200}?)\.", flags=re.IGNORECASE
    )

    for m in sentence_pattern.finditer(content):
        start, end = m.span()
        sentence = m.group(0)

        if not re.search(r"opt(?:\\-)?level", sentence, flags=re.IGNORECASE):
            continue

        num_match = re.search(r"(\d+)", sentence)
        if not num_match:
            continue
        old_level = int(num_match.group(1))

        window_start = max(0, start - 1200)
        window = content[window_start:start]

        # Use any() for better readability
        if not any(
            [
                re.search(r"(^|\n)\s*-O\b", window),
                re.search(r"\\fB\\-?O\\fR", window),
                re.search(r"\\-O\b", window),
                re.search(r"\.B\s+-O\b", window),
                re.search(r"\\fB-?O\\fP", window),
            ]
        ):
            continue

        # We found at least one -O entry with opt-level
        found_patterns = True

        if old_level == new_level:
            continue

        # More robust: replace only the number after opt-level=
        new_sentence = re.sub(
            r"(opt(?:\\-)?level\s*=\s*)\d+",
            rf"\g<1>{new_level}",
            sentence,
            count=1,
        )
        out_parts.append(content[last_index:start])
        out_parts.append(new_sentence)
        last_index = end
        replacements += 1

    out_parts.append(content[last_index:])
    return "".join(out_parts), replacements, found_patterns


def show_colored_diff(old: str, new: str, filename: str, color: bool) -> None:
    # Use keepends=False and lineterm="\n" for consistency
    old_lines = old.splitlines(keepends=False)
    new_lines = new.splitlines(keepends=False)
    diff_iter = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=filename,
        tofile=filename + " (updated)",
        lineterm="\n",
    )
    for line in diff_iter:
        if line.startswith("---") or line.startswith("+++"):
            print(colorize(line, "bold", color))
        elif line.startswith("@@"):
            print(colorize(line, "yellow", color))
        elif line.startswith("+"):
            print(colorize(line, "green", color))
        elif line.startswith("-"):
            print(colorize(line, "red", color))
        else:
            print(line)


def backup_file(path: Path) -> Path:
    # Added microseconds to avoid collision if run multiple times
    # per second
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")
    backup = path.with_name(path.name + f".bak.{ts}")
    shutil.copy2(path, backup)
    return backup


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Update rustc man page -O opt-level to match rustc --help")
    )
    p.add_argument(
        "--man-file",
        "-m",
        required=True,
        help=("Path to rustc man page file to update (e.g. src/doc/man/rustc.1)"),
    )
    p.add_argument(
        "--rustc-cmd",
        default=DEFAULT_RUSTC,
        help="rustc binary to query (default: rustc)",
    )
    p.add_argument(
        "--expected-level",
        "-e",
        type=int,
        help="Use this level instead of querying rustc",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Write changes to the man file (creates a backup). "
            "Without this flag runs in dry-run mode and prints "
            "a diff."
        ),
    )
    p.add_argument("--no-color", action="store_true", help="Disable colored output")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    color = (not args.no_color) and sys.stdout.isatty()

    man_path = Path(args.man_file)
    if not man_path.exists():
        print(f"Error: man file not found: {man_path}", file=sys.stderr)
        return 2

    try:
        new_level = (
            args.expected_level
            if args.expected_level is not None
            else get_rustc_opt_level(args.rustc_cmd)
        )
    except RuntimeError as e:
        print(f"Error determining rustc opt-level: {e}", file=sys.stderr)
        return 3

    try:
        content = man_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading man file {man_path}: {e}", file=sys.stderr)
        return 4

    new_content, replacements, found_patterns = find_and_replace_manpage_content(
        content, new_level
    )

    if replacements == 0:
        if found_patterns:
            print(f"✓ Manpage is already up-to-date (opt-level={new_level}).")
        else:
            print(
                "Warning: Could not find -O entry with "
                "'Equivalent to -C opt-level=' pattern in manpage.",
                file=sys.stderr,
            )
            return 6
        return 0

    header = f"Found {replacements} replacement(s). Proposed changes:"
    print(colorize(header, "bold", color))
    show_colored_diff(content, new_content, str(man_path), color)

    if args.apply:
        try:
            backup = backup_file(man_path)
            man_path.write_text(new_content, encoding="utf-8")
            msg = f"\nApplied changes to {man_path}. Backup saved to {backup}"
            print(colorize(msg, "green", color))
        except Exception as e:
            print(f"Error writing updated man file: {e}", file=sys.stderr)
            return 5
    else:
        print("\nDry-run only. Use --apply to write changes to disk.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
