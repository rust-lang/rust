#!/bin/bash

set -eu
source shared.sh

if [ -z "$1" ]; then
    echo "Usage: ${0} <commit_id>"
    exit -1
fi

target="x86_64-fortanix-unknown-sgx"
url="https://github.com/fortanix/llvm-project/archive/${1}.tar.gz"
repo_name="llvm-project"

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
