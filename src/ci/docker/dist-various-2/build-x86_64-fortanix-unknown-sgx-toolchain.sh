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

install_prereq()
{
    apt-get update
    apt-get install -y --no-install-recommends \
            build-essential \
            ca-certificates \
            cmake \
            git
}

# Clone Fortanix's port of llvm-project to build libunwind that would link with this target.
# The below method to download a single commit from llvm-project is based on fetch_submodule
# from init_repo.sh
fetch_llvm_commit()
{
    cached="download-${repo_name}.tar.gz"
    curl -f -sSL -o ${cached} ${url}
    tar -xvzf ${cached}
    mkdir "./${repo_name}" && tar -xf ${cached} -C ${repo_name} --strip-components 1
}

build_unwind()
{
    dir_name="${target}_temp"
    rm -rf "./${dir_name}"
    mkdir -p ${dir_name}
    cd ${dir_name}

    retry fetch_llvm_commit
    cd "${repo_name}/libunwind"

    # Build libunwind
    mkdir -p build
    cd build
    cmake -DCMAKE_BUILD_TYPE="RELEASE" -DRUST_SGX=1 -G "Unix Makefiles" -DLLVM_PATH=../../llvm/ ../
    make unwind_static
    install -D "lib/libunwind.a" "/${target}/lib/libunwind.a"
    rm -rf ${dir_name}
}

set -x
hide_output install_prereq
hide_output build_unwind
