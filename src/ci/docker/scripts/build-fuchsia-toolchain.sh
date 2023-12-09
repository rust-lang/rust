#!/usr/bin/env bash

set -ex
source shared.sh

FUCHSIA_SDK_URL=https://chrome-infra-packages.appspot.com/dl/fuchsia/sdk/core/linux-amd64
FUCHSIA_SDK_ID=MrhQwtmP8CpZre-i_PNOREcThbUcrX3bA-45d6WQr-cC
FUCHSIA_SDK_SHA256=32b850c2d98ff02a59adefa2fcf34e44471385b51cad7ddb03ee3977a590afe7
FUCHSIA_SDK_USR_DIR=/usr/local/core-linux-amd64-fuchsia-sdk
CLANG_DOWNLOAD_URL=\
https://chrome-infra-packages.appspot.com/dl/fuchsia/third_party/clang/linux-amd64
CLANG_DOWNLOAD_ID=Tpc85d1ZwSlZ6UKl2d96GRUBGNA5JKholOKe24sRDr0C
CLANG_DOWNLOAD_SHA256=4e973ce5dd59c12959e942a5d9df7a19150118d03924a86894e29edb8b110ebd

install_clang() {
  mkdir -p clang_download
  pushd clang_download > /dev/null

  # Download clang+llvm
  curl -LO "${CLANG_DOWNLOAD_URL}/+/${CLANG_DOWNLOAD_ID}"
  echo "$(echo ${CLANG_DOWNLOAD_SHA256}) ${CLANG_DOWNLOAD_ID}" | sha256sum --check --status
  unzip -qq ${CLANG_DOWNLOAD_ID} -d clang-linux-amd64

  # Other dists currently depend on our Clang... moving into /usr/local for other
  #  dist usage instead of a Fuchsia /usr/local directory
  chmod -R 777 clang-linux-amd64/.
  cp -a clang-linux-amd64/. /usr/local

  # CFLAGS and CXXFLAGS env variables in main Dockerfile handle sysroot linking
  for arch in x86_64 aarch64; do
    for tool in clang clang++; do
      ln -s /usr/local/bin/${tool} /usr/local/bin/${arch}-unknown-fuchsia-${tool}
    done
    ln -s /usr/local/bin/llvm-ar /usr/local/bin/${arch}-unknown-fuchsia-ar
  done

  popd > /dev/null
  rm -rf clang_download
}

install_zircon_libs() {
  mkdir -p zircon
  pushd zircon > /dev/null

  # Download Fuchsia SDK (with Zircon libs)
  curl -LO "${FUCHSIA_SDK_URL}/+/${FUCHSIA_SDK_ID}"
  echo "$(echo ${FUCHSIA_SDK_SHA256}) ${FUCHSIA_SDK_ID}" | sha256sum --check --status
  unzip -qq ${FUCHSIA_SDK_ID} -d core-linux-amd64

  # Moving SDK into Docker's user-space
  mkdir -p ${FUCHSIA_SDK_USR_DIR}
  chmod -R 777 core-linux-amd64/.
  cp -r core-linux-amd64/* ${FUCHSIA_SDK_USR_DIR}

  popd > /dev/null
  rm -rf zircon
}

hide_output install_clang
hide_output install_zircon_libs
