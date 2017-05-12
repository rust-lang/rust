#!/bin/bash
# Copyright 2017 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# ignore-tidy-linelength

set -ex
source shared.sh

# Download sources
SRCS=(
  "https://fuchsia.googlesource.com/magenta magenta d17073dc8de344ead3b65e8cc6a12280dec38c84"
  "https://llvm.googlesource.com/llvm llvm 3f58a16d8eec385e2b3ebdfbb84ff9d3bf27e025"
  "https://llvm.googlesource.com/clang llvm/tools/clang 727ea63e6e82677f6e10e05e08bc7d6bdbae3111"
  "https://llvm.googlesource.com/lld llvm/tools/lld a31286c1366e5e89b8872803fded13805a1a084b"
  "https://llvm.googlesource.com/lldb llvm/tools/lldb 0b2384abec4cb99ad66687712e07dee4dd9d187e"
  "https://llvm.googlesource.com/compiler-rt llvm/runtimes/compiler-rt 9093a35c599fe41278606a20b51095ea8bd5a081"
  "https://llvm.googlesource.com/libcxx llvm/runtimes/libcxx 607e0c71ec4f7fd377ad3f6c47b08dbe89f66eaa"
  "https://llvm.googlesource.com/libcxxabi llvm/runtimes/libcxxabi 0a3a1a8a5ca5ef69e0f6b7d5b9d13e63e6fd2c19"
  "https://llvm.googlesource.com/libunwind llvm/runtimes/libunwind e128003563d99d9ee62247c4cee40f07d21c03e3"
)

fetch() {
  mkdir -p $2
  pushd $2 > /dev/null
  git init
  git remote add origin $1
  git fetch --depth=1 origin $3
  git reset --hard FETCH_HEAD
  popd > /dev/null
}

for i in "${SRCS[@]}"; do
  fetch $i
done

# Remove this once https://reviews.llvm.org/D28791 is resolved
cd llvm/runtimes/compiler-rt
patch -Np1 < /tmp/compiler-rt-dso-handle.patch
cd ../../..

# Build toolchain
cd llvm
mkdir build
cd build
hide_output cmake -GNinja \
  -DFUCHSIA_SYSROOT=${PWD}/../../magenta/third_party/ulib/musl \
  -DLLVM_ENABLE_LTO=OFF \
  -DCLANG_BOOTSTRAP_PASSTHROUGH=LLVM_ENABLE_LTO \
  -C ../tools/clang/cmake/caches/Fuchsia.cmake \
  ..
hide_output ninja stage2-distribution
hide_output ninja stage2-install-distribution
cd ../..

# Build sysroot
rm -rf llvm/runtimes/compiler-rt
./magenta/scripts/download-toolchain

build_sysroot() {
  local arch="$1"

  case "${arch}" in
    x86_64) tgt="magenta-pc-x86-64" ;;
    aarch64) tgt="magenta-qemu-arm64" ;;
  esac

  hide_output make -C magenta -j$(getconf _NPROCESSORS_ONLN) $tgt
  dst=/usr/local/${arch}-unknown-fuchsia
  mkdir -p $dst
  cp -r magenta/build-${tgt}/sysroot/include $dst/
  cp -r magenta/build-${tgt}/sysroot/lib $dst/

  cd llvm
  mkdir build-runtimes-${arch}
  cd build-runtimes-${arch}
  hide_output cmake -GNinja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_AR=/usr/local/bin/llvm-ar \
    -DCMAKE_RANLIB=/usr/local/bin/llvm-ranlib \
    -DCMAKE_INSTALL_PREFIX= \
    -DLLVM_MAIN_SRC_DIR=${PWD}/.. \
    -DLLVM_BINARY_DIR=${PWD}/../build \
    -DLLVM_ENABLE_WERROR=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_INCLUDE_TESTS=ON \
    -DCMAKE_SYSTEM_NAME=Fuchsia \
    -DCMAKE_C_COMPILER_TARGET=${arch}-fuchsia \
    -DCMAKE_CXX_COMPILER_TARGET=${arch}-fuchsia \
    -DUNIX=1 \
    -DLIBCXX_HAS_MUSL_LIBC=ON \
    -DLIBCXXABI_USE_LLVM_UNWINDER=ON \
    -DCMAKE_SYSROOT=${dst} \
    -DCMAKE_C_COMPILER_FORCED=TRUE \
    -DCMAKE_CXX_COMPILER_FORCED=TRUE \
    -DLLVM_ENABLE_LIBCXX=ON \
    -DCMAKE_EXE_LINKER_FLAGS="-nodefaultlibs -lc" \
    -DCMAKE_SHARED_LINKER_FLAGS="$(clang --target=${arch}-fuchsia -print-libgcc-file-name)" \
    ../runtimes
  hide_output env DESTDIR="${dst}" ninja install
  cd ../..
}

build_sysroot "x86_64"
build_sysroot "aarch64"

rm -rf magenta llvm

for arch in x86_64 aarch64; do
  for tool in clang clang++; do
    cat >/usr/local/bin/${arch}-unknown-fuchsia-${tool} <<EOF
#!/bin/sh
${tool} --target=${arch}-unknown-fuchsia --sysroot=/usr/local/${arch}-unknown-fuchsia "\$@"
EOF
    chmod +x /usr/local/bin/${arch}-unknown-fuchsia-${tool}
  done
  ln -s /usr/local/bin/llvm-ar /usr/local/bin/${arch}-unknown-fuchsia-ar
done
