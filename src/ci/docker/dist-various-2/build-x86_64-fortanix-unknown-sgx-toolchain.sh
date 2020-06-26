#!/bin/bash

set -eu
source shared.sh

target="x86_64-fortanix-unknown-sgx"

install_prereq() {
    curl https://apt.llvm.org/llvm-snapshot.gpg.key|apt-key add -
    add-apt-repository -y 'deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic main'
    apt-get update
    apt-get install -y --no-install-recommends \
            build-essential \
            ca-certificates \
            cmake \
            git \
            clang-11
}

hide_output install_prereq
