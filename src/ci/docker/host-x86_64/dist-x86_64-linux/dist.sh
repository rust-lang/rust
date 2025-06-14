#!/bin/bash

set -eux

python3 ../x.py build --set rust.debug=true opt-dist

./build/$HOSTS/stage0-tools-bin/opt-dist linux-ci -- python3 ../x.py dist \
    --host $HOSTS --target $HOSTS \
    --include-default-paths \
    build-manifest bootstrap

# Use GCC for building GCC, as it seems to behave badly when built with Clang
CC=/rustroot/bin/cc CXX=/rustroot/bin/c++ python3 ../x.py dist gcc
