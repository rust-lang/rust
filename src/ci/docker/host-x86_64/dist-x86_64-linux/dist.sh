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
    # We have to build our own binutils for the GCC build, because the default CentOS 7 binutils are
    # too old, and they do not support `SHF_GNU_RETAIN`.
    BINUTILS="2.47"
    BINUTILS_ROOT_PATH="$(pwd)/binutils-install"
    export BINUTILS_PATH="$BINUTILS_ROOT_PATH/bin"
    curl https://ci-mirrors.rust-lang.org/rustc/gcc/binutils-$BINUTILS.tar.xz | xzcat | tar xf -
    mkdir binutils-build
    mkdir "$BINUTILS_ROOT_PATH"
    cd binutils-build
    hide_output ../binutils-$BINUTILS/configure --prefix="$BINUTILS_ROOT_PATH"
    hide_output make -j$(nproc)
    hide_output make install

    cd ..
    rm -rf binutils-build binutils-$BINUTILS

    if echo '.section .test,"awR",@progbits' | "$BINUTILS_PATH"/as - -o /dev/null 2>/dev/null; then
        echo "binutils assembler supports SHF_GNU_RETAIN"
    else
        echo "binutils assembler DOES NOT support SHF_GNU_RETAIN"
        exit 1
    fi

    PATH="$BINUTILS_PATH":$PATH CC=/rustroot/bin/cc CXX=/rustroot/bin/c++ python3 ../x.py dist \
      gcc-dev \
      gcc
fi
