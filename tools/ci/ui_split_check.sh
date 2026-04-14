#!/usr/bin/env bash
set -euo pipefail

fail=0

# Bloom must not link SVG/font/text raster modules.
if rg -n "fontdue|serde|libm|spin" userspace/bloom/Cargo.toml >/dev/null; then
  echo "ui-split: Bloom Cargo.toml contains forbidden deps" >&2
  fail=1
fi

if rg -n "\bmod\s+(svg|raster|drawlist|lowered|font_graph|ui)\b" userspace/bloom/src/main.rs >/dev/null; then
  echo "ui-split: Bloom main.rs still declares paint modules" >&2
  fail=1
fi

if rg -n "SVG|svg::|raster::|DrawCmd|PaintObject|font_graph" userspace/bloom/src \
  --glob '!*.svg' >/dev/null; then
  echo "ui-split: Bloom sources reference paint/svg/text modules" >&2
  fail=1
fi

# Only Blossom should write snapshot keys.
if rg -n "prop_set\([^\)]*(UI_SNAPSHOT_BYTESPACE|UI_SNAPSHOT_WIDTH|UI_SNAPSHOT_HEIGHT|UI_SNAPSHOT_STRIDE|UI_SNAPSHOT_FORMAT|UI_PRESENT_EPOCH)" \
  userspace --glob '*.rs' --glob '!userspace/blossom/**' >/dev/null; then
  echo "ui-split: snapshot keys written outside Blossom" >&2
  fail=1
fi

exit $fail
