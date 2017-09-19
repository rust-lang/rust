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
  "https://fuchsia.googlesource.com/zircon zircon e9a26dbc70d631029f8ee9763103910b7e3a2fe1"
  "https://llvm.googlesource.com/llvm llvm 65bdf0ae4a87e6992c24f06e2612909952468710"
  "https://llvm.googlesource.com/clang llvm/tools/clang 914987de45cf83636537909ce09156aa7a37d6ec"
  "https://llvm.googlesource.com/clang-tools-extra llvm/tools/clang/tools/extra 83de24124250a7cdc7a0fdc61b7e3c3d64b80225"
  "https://llvm.googlesource.com/lld llvm/tools/lld f8ed4483c589b390daafac92e28f4680ad052643"
  "https://llvm.googlesource.com/lldb llvm/tools/lldb 55cf8753321782668cb7e2d879457ee1ad57a2b9"
  "https://llvm.googlesource.com/compiler-rt llvm/runtimes/compiler-rt a8682fdf74d3cb93769b7394f2cdffc5cefb8bd8"
  "https://llvm.googlesource.com/libcxx llvm/runtimes/libcxx 5f919fe349450b3da0e29611ae37f6a940179290"
  "https://llvm.googlesource.com/libcxxabi llvm/runtimes/libcxxabi caa78daf9285dada17e3e6b8aebcf7d128427f83"
  "https://llvm.googlesource.com/libunwind llvm/runtimes/libunwind 469bacd2ea64679c15bb4d86adf000f2f2c27328"
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

# Build sysroot
./zircon/scripts/download-toolchain

build_sysroot() {
  local arch="$1"

  case "${arch}" in
    x86_64) tgt="zircon-pc-x86-64" ;;
    aarch64) tgt="zircon-qemu-arm64" ;;
  esac

  hide_output make -C zircon -j$(getconf _NPROCESSORS_ONLN) $tgt
  dst=/usr/local/${arch}-unknown-fuchsia
  mkdir -p $dst
  cp -r zircon/build-${tgt}/sysroot/include $dst/
  cp -r zircon/build-${tgt}/sysroot/lib $dst/
}

for arch in x86_64 aarch64; do
  build_sysroot ${arch}
done

# Build toolchain
cd llvm
mkdir build
cd build
hide_output cmake -GNinja \
  -DFUCHSIA_x86_64_SYSROOT=/usr/local/x86_64-unknown-fuchsia \
  -DFUCHSIA_aarch64_SYSROOT=/usr/local/aarch64-unknown-fuchsia \
  -DLLVM_ENABLE_LTO=OFF \
  -DCLANG_BOOTSTRAP_PASSTHROUGH=LLVM_ENABLE_LTO \
  -C ../tools/clang/cmake/caches/Fuchsia.cmake \
  ..
hide_output ninja stage2-distribution
hide_output ninja stage2-install-distribution
cd ../..

rm -rf zircon llvm

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
