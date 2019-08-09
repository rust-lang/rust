#!/usr/bin/env bash

set -ex
source shared.sh

ZIRCON=e9a26dbc70d631029f8ee9763103910b7e3a2fe1

mkdir -p zircon
pushd zircon > /dev/null

# Download sources
git init
git remote add origin https://github.com/rust-lang-nursery/mirror-google-fuchsia-zircon
git fetch --depth=1 origin $ZIRCON
git reset --hard FETCH_HEAD

# Download toolchain
./scripts/download-toolchain
chmod -R a+rx prebuilt/downloads/clang+llvm-x86_64-linux
cp -a prebuilt/downloads/clang+llvm-x86_64-linux/. /usr/local

build() {
  local arch="$1"

  case "${arch}" in
    x86_64) tgt="zircon-pc-x86-64" ;;
    aarch64) tgt="zircon-qemu-arm64" ;;
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
