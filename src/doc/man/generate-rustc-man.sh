#!/usr/bin/env sh
set -eu

if [ $# -ne 1 ]; then
    echo "usage: $0 /path/to/rustc" >&2
    exit 1
fi

RUSTC="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
OUT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Generating rustc.1 man page with $RUSTC binary and placing it in $OUT_DIR"

help2man --help-option="--help -v" --no-discard-stderr --name="Rust language compiler" \
    "$RUSTC" -o "$OUT_DIR/rustc.1"

echo "Man page generated: $OUT_DIR/rustc.1"
