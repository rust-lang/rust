#!/usr/bin/env bash
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

ZIRCON=1e827755b8b48bb5029e3c7cf2ea9ced9c8f8e1d

mkdir -p zircon
pushd zircon > /dev/null

# Download sources
git init
git remote add origin https://fuchsia.googlesource.com/zircon
git fetch --depth=1 origin $ZIRCON
git reset --hard FETCH_HEAD

# Download toolchain
./scripts/download-prebuilt
chmod -R a+rx prebuilt/downloads/clang
cp -a prebuilt/downloads/clang/. /usr/local
if ! which python > /dev/null; then
  ln -s `which python2.7` /usr/local/bin/python
fi

build() {
  local arch="$1"

  case "${arch}" in
    x86_64) tgt="x64" ;;
    aarch64) tgt="arm64" ;;
  esac

  hide_output make -j$(getconf _NPROCESSORS_ONLN) $tgt
  dst=/usr/local/${arch}-fuchsia
  mkdir -p $dst
  cp -a build-${tgt}/sysroot/include $dst/
  cp -a build-${tgt}/sysroot/lib $dst/
}

# Build sysroot
for arch in x86_64 aarch64; do
  build ${arch}
done

popd > /dev/null
rm -rf zircon

for arch in x86_64 aarch64; do
  for tool in clang clang++; do
    cat >/usr/local/bin/${arch}-fuchsia-${tool} <<EOF
#!/bin/sh
${tool} --target=${arch}-fuchsia --sysroot=/usr/local/${arch}-fuchsia "\$@"
EOF
    chmod +x /usr/local/bin/${arch}-fuchsia-${tool}
  done
  ln -s /usr/local/bin/llvm-ar /usr/local/bin/${arch}-fuchsia-ar
done
