#!/bin/bash
set -e

source scripts/ext_config.sh

./build.sh --without-sysroot "$@"

rm -r target/out || true

scripts/tests.sh no_sysroot

./build.sh "$@"

scripts/tests.sh base_sysroot
scripts/tests.sh extended_sysroot
