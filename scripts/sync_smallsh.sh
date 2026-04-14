#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_URL="https://github.com/loganintech/smallsh.git"
UPSTREAM_COMMIT="8715f2d29ebaf05bdd3d136e36417da6141c15c3"
DEST_DIR="$ROOT_DIR/userspace/smallsh/upstream"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "==> Fetching smallsh at $UPSTREAM_COMMIT"
git clone --depth 1 --filter=blob:none --no-checkout "$UPSTREAM_URL" "$TMP_DIR/repo"
git -C "$TMP_DIR/repo" fetch --depth 1 origin "$UPSTREAM_COMMIT"
git -C "$TMP_DIR/repo" checkout "$UPSTREAM_COMMIT"

rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR/src/builtins" "$DEST_DIR/src/process_pool"

cp "$TMP_DIR/repo/Cargo.toml" "$DEST_DIR/Cargo.toml"
cp "$TMP_DIR/repo/README.md" "$DEST_DIR/README.md"
cp "$TMP_DIR/repo/src/main.rs" "$DEST_DIR/src/main.rs"
cp "$TMP_DIR/repo/src/builtins/"*.rs "$DEST_DIR/src/builtins/"
cp "$TMP_DIR/repo/src/process_pool/"*.rs "$DEST_DIR/src/process_pool/"

echo "==> Refreshed $DEST_DIR from $UPSTREAM_URL@$UPSTREAM_COMMIT"
