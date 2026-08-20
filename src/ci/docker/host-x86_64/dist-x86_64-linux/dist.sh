#!/bin/bash

set -eux

python3 ../x.py build --set rust.debug=true opt-dist

./build/$HOSTS/stage1-tools-bin/opt-dist linux-ci -- python3 ../x.py dist \
    --host $HOSTS --target $HOSTS \
    --include-default-paths \
    build-manifest \
    bootstrap \
    enzyme \
    offload \
    rustc_codegen_gcc

# Use GCC for building GCC components, as it seems to behave badly when built with Clang
# Only build GCC on full builds, not try builds
if [ "${DIST_TRY_BUILD:-0}" == "0" ]; then
    # We add the binutils binary path to ensure that it uses the 2.47 version of binutils,
    # which is needed for the `retain` attribute feature.
    export PATH="/tmp/binutils-install/bin:$PATH"

    CC=/rustroot/bin/cc CXX=/rustroot/bin/c++ \
        python3 ../x.py dist \
        gcc-dev \
        gcc

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
