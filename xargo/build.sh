#!/bin/sh
cd "$(dirname "$0")"
# The flags here should be kept in sync with `add_miri_default_args` in `src/lib.rs`.
RUSTFLAGS='-Zalways-encode-mir -Zmir-emit-retag -Zmir-opt-level=0' xargo build
