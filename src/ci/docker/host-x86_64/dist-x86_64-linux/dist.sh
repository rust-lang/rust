#!/bin/bash

set -eux

# TESTING-ONLY CHANGES.
touch github-summary.md
python3 ../x.py dist \
    --host $HOSTS --target $HOSTS \
    --set "rust.codegen-backends=['llvm']" \
    rustc rust-std cargo rust-src

# Use GCC for building GCC components, as it seems to behave badly when built with Clang
# Only build GCC on full builds, not try builds
if [ "${DIST_TRY_BUILD:-0}" == "0" ]; then
    CC=/rustroot/bin/cc CXX=/rustroot/bin/c++ python3 ../x.py dist \
      gcc-dev \
      gcc
fi
