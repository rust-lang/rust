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

build_unwind() {
    set -x
    dir_name="${target}_temp"
    rm -rf ${dir_name}
    mkdir -p ${dir_name}
    pushd ${dir_name}

    # Clone Fortanix's fork of llvm-project which has a port of libunwind
    fetch_github_commit_archive "$repo_name" "$url"
    cd "${repo_name}/libunwind"

    # Build libunwind
    mkdir -p build
    cd build
    target_CC="CC_${target//-/_}"
    target_CXX="CXX_${target//-/_}"
    target_CFLAGS="CFLAGS_${target//-/_}"
    target_CXXFLAGS="CXXFLAGS_${target//-/_}"
    cmake -DCMAKE_BUILD_TYPE="RELEASE" -DRUST_SGX=1 -G "Unix Makefiles" \
        -DCMAKE_C_COMPILER="${!target_CC}" -DCMAKE_CXX_COMPILER="${!target_CXX}" \
        -DCMAKE_C_FLAGS="${!target_CFLAGS}" -DCMAKE_CXX_FLAGS="${!target_CXXFLAGS}" \
        -DCMAKE_C_COMPILER_TARGET=$target -DCMAKE_CXX_COMPILER_TARGET=$target \
        -DLLVM_ENABLE_WARNINGS=1 -DLIBUNWIND_ENABLE_WERROR=1 -DLIBUNWIND_ENABLE_PEDANTIC=0 \
        -DLLVM_PATH=../../llvm/ ../
    make unwind_static
    install -D "lib/libunwind.a" "/${target}/lib/libunwind.a"

    popd
    rm -rf ${dir_name}

    { set +x; } 2>/dev/null
}

set -x
hide_output install_prereq
build_unwind
