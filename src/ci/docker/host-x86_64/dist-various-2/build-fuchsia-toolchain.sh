#!/usr/bin/env bash

set -ex
source shared.sh

FUCHSIA_SDK_URL=https://chrome-infra-packages.appspot.com/dl/fuchsia/sdk/core/linux-amd64
FUCHSIA_SDK_ID=4xjxrGUrDbQ6_zJwj6cDN1IbWsWV5aCQXC_zO_Hu0XkC
FUCHSIA_SDK_SHA256=e318f1ac652b0db43aff32708fa70337521b5ac595e5a0905c2ff33bf1eed179
FUCHSIA_SDK_USR_DIR=/usr/local/core-linux-amd64-fuchsia-sdk
CLANG_DOWNLOAD_URL=\
https://chrome-infra-packages.appspot.com/dl/fuchsia/third_party/clang/linux-amd64
CLANG_DOWNLOAD_ID=vU0vNjSihOV4Q6taQYCpy03JXGiCyVwxen3rFMNMIgsC
CLANG_DOWNLOAD_SHA256=bd4d2f3634a284e57843ab5a4180a9cb4dc95c6882c95c317a7deb14c34c220b

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
