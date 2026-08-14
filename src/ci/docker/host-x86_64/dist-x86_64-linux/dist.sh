#!/bin/bash

set -eux

python3 ../x.py build --set rust.debug=true opt-dist

./build/$HOSTS/stage1-tools-bin/opt-dist linux-ci -- python3 ../x.py dist \
    --host $HOSTS --target $HOSTS \
    --include-default-paths \
    build-manifest \
    bootstrap \
    enzyme \
    rustc_codegen_gcc

# Use GCC for building GCC components, as it seems to behave badly when built with Clang
# Only build GCC on full builds, not try builds
if [ "${DIST_TRY_BUILD:-0}" == "0" ]; then
    CC=/rustroot/bin/cc CXX=/rustroot/bin/c++ python3 ../x.py dist \
      gcc-dev \
      gcc
    # We confirm that the built GCC has support for the `retain` attribute.
    # FIXME: Maybe get the path from `.x.py` instead?
    gcc_path="./build/$HOSTS/gcc/$HOSTS/install/bin/gcc"
    if echo 'int x __attribute__((used, retain));' | "$gcc_path" -S -x c -o - - | grep -i '"a.*R"'; then
        echo "retain attribute is supported"
    else
        echo "retain attribute is not supported"
        # We display the generated asm just in case...
        echo 'int x __attribute__((used, retain));' | "$gcc_path" -S -x c -o - -
        exit 1
    fi
fi
