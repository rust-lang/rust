#!/usr/bin/env bash

set -ex
source shared.sh

FUCHSIA_SDK_URL=https://chrome-infra-packages.appspot.com/dl/fuchsia/sdk/core/linux-amd64
FUCHSIA_SDK_ID=version:26.20241211.7.1
FUCHSIA_SDK_SHA256=2cb7a9a0419f7413a46e0ccef7dad89f7c9979940d7c1ee87fac70ff499757d6
FUCHSIA_SDK_USR_DIR=/usr/local/core-linux-amd64-fuchsia-sdk
CLANG_DOWNLOAD_URL=\
https://chrome-infra-packages.appspot.com/dl/fuchsia/third_party/clang/linux-amd64
CLANG_DOWNLOAD_ID=git_revision:388d7f144880dcd85ff31f06793304405a9f44b6
CLANG_DOWNLOAD_SHA256=970d1f427b9c9a3049d8622c80c86830ff31b5334ad8da47a2f1e81143197e8b

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
