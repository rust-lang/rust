#!/usr/bin/env bash

set -ex
source shared.sh

# The Fuchsia SDK is used to set up a Fuchsia test environment. The latest version can be found at:
# the latest Fuchsia checkout is found at:
#   https://chrome-infra-packages.appspot.com/p/fuchsia/sdk/core/linux-amd64/+/latest
FUCHSIA_SDK_URL=https://chrome-infra-packages.appspot.com/dl/fuchsia/sdk/core/linux-amd64
FUCHSIA_SDK_ID=version:27.20250319.0.1
FUCHSIA_SDK_SHA256=0b0eeed62b024d8910917a797d658098a312b7674c3742292b40ba10aa56ea8e
FUCHSIA_SDK_USR_DIR=/usr/local/core-linux-amd64-fuchsia-sdk

# The Clang toolchain used to compile Fuchsia. The version currently used by
# the latest Fuchsia checkout is found in:
#   https://fuchsia.googlesource.com/integration/+/refs/heads/main/toolchain. The
# CIPD artifacts are found in:
#   https://chrome-infra-packages.appspot.com/p/fuchsia/third_party/clang/linux-amd64
CLANG_DOWNLOAD_URL=\
https://chrome-infra-packages.appspot.com/dl/fuchsia/third_party/clang/linux-amd64
CLANG_DOWNLOAD_ID=git_revision:684052173971868aab0e6b62d7770a6299e84141
CLANG_DOWNLOAD_SHA256=e82b7f96e1215d68fb8d39f21fdc5020ee4683baae247ac553017f65819bc409

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
