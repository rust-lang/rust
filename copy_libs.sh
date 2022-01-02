#!/bin/bash

# FIXME temporarily add this to Rust CI instead of in a separate shell file

export GCC_LIBS_HACK="$(cygpath -am missing-libs-hack)"
MINGW_PREFIX="/clang64"
mkdir -p "${GCC_LIBS_HACK}"
cp "$(cygpath -u $(clang -print-libgcc-file-name))" "${GCC_LIBS_HACK}/libgcc.a"
cp "${MINGW_PREFIX}/lib/libunwind.a" "${GCC_LIBS_HACK}/libgcc_eh.a"
cp "${MINGW_PREFIX}/lib/libunwind.dll.a" "${GCC_LIBS_HACK}/libgcc_s.a"

# Run this in your terminal after running this script
# export RUSTFLAGS_BOOTSTRAP="-C link-arg=-L$(cygpath -am missing-libs-hack)"
