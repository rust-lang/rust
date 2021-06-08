#!/usr/bin/env bash
set -e

./y.rs build --sysroot none "$@"

rm -r target/out || true

scripts/tests.sh no_sysroot

./y.rs build "$@"

scripts/tests.sh base_sysroot
scripts/tests.sh extended_sysroot
