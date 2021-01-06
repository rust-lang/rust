#!/bin/bash
set -e

export CG_CLIF_DISPLAY_CG_TIME=1

./build.sh --without-sysroot "$@"

rm -r target/out || true

scripts/tests.sh no_sysroot

./build.sh "$@"

scripts/tests.sh base_sysroot
scripts/tests.sh extended_sysroot
