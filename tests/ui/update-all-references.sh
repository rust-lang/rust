#!/bin/bash

# A script to update the references for all tests. The idea is that
# you do a run, which will generate files in the build directory
# containing the (normalized) actual output of the compiler. You then
# run this script, which will copy those files over. If you find
# yourself manually editing a foo.stderr file, you're doing it wrong.
#
# See all `update-references.sh`, if you just want to update a single test.

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "usage: $0"
fi

CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target}
PROFILE=${PROFILE:-debug}
BUILD_DIR=${CARGO_TARGET_DIR}/${PROFILE}/test_build_base

MY_DIR=$(dirname "$0")
cd "$MY_DIR" || exit
find . -name '*.rs' -exec ./update-references.sh "$BUILD_DIR" {} +
