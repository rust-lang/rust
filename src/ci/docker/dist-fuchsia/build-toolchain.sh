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

set -ex
source shared.sh

# Download sources
SRCS=(
  "https://fuchsia.googlesource.com/magenta magenta ac69119"
  "https://fuchsia.googlesource.com/third_party/llvm llvm 5463083"
  "https://fuchsia.googlesource.com/third_party/clang llvm/tools/clang 4ff7b4b"
  "https://fuchsia.googlesource.com/third_party/lld llvm/tools/lld fd465a3"
  "https://fuchsia.googlesource.com/third_party/lldb llvm/tools/lldb 6bb11f8"
  "https://fuchsia.googlesource.com/third_party/compiler-rt llvm/runtimes/compiler-rt 52d4ecc"
  "https://fuchsia.googlesource.com/third_party/libcxx llvm/runtimes/libcxx e891cc8"
  "https://fuchsia.googlesource.com/third_party/libcxxabi llvm/runtimes/libcxxabi f0f0257"
  "https://fuchsia.googlesource.com/third_party/libunwind llvm/runtimes/libunwind 50bddc1"
)

fetch() {
  mkdir -p $2
  pushd $2 > /dev/null
  curl -sL $1/+archive/$3.tar.gz | tar xzf -
  popd > /dev/null
}

for i in "${SRCS[@]}"; do
  fetch $i
done

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
