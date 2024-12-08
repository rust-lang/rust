#!/usr/bin/env bash

set -ex
source shared.sh

FUCHSIA_SDK_URL=https://chrome-infra-packages.appspot.com/dl/fuchsia/sdk/core/linux-amd64
FUCHSIA_SDK_ID=version:21.20240610.2.1
FUCHSIA_SDK_SHA256=2d2d057fc3f0404197cced2200f88cbcdaaf5fbf6475955045091f8676791ce7
FUCHSIA_SDK_USR_DIR=/usr/local/core-linux-amd64-fuchsia-sdk
CLANG_DOWNLOAD_URL=\
https://chrome-infra-packages.appspot.com/dl/fuchsia/third_party/clang/linux-amd64
CLANG_DOWNLOAD_ID=git_revision:3809e20afc68d7d03821f0ec59b928dcf9befbf4
CLANG_DOWNLOAD_SHA256=3c2c442b61cd9e8f1b567738f6d53cffe11b3fc820e7dae87a82a0859be8f204

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
