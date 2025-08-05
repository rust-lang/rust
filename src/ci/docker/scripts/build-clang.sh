#!/usr/bin/env bash

set -ex

source shared.sh

# Try to keep the LLVM version here in sync with src/ci/scripts/install-clang.sh
LLVM=llvmorg-21.1.0-rc2

mkdir llvm-project
cd llvm-project

curl -L https://github.com/llvm/llvm-project/archive/$LLVM.tar.gz | \
  tar xzf - --strip-components=1

mkdir clang-build
cd clang-build

# For whatever reason the default set of include paths for clang is different
# than that of gcc. As a result we need to manually include our sysroot's
# include path, /rustroot/include, to clang's default include path.
INC="/rustroot/include:/usr/include"

GCC_PLUGIN_TARGET=$GCC_BUILD_TARGET
# We build gcc for the i686 job on x86_64 so the plugin will end up under an x86_64 path
if [[ $GCC_PLUGIN_TARGET == "i686-pc-linux-gnu" ]]; then
  GCC_PLUGIN_TARGET=x86_64-pc-linux-gnu
fi

# We need compiler-rt for the profile runtime (used later to PGO the LLVM build)
# but sanitizers aren't currently building. Since we don't need those, just
# disable them. BOLT is used for optimizing LLVM.
hide_output \
    cmake ../llvm \
      -DCMAKE_C_COMPILER=/rustroot/bin/gcc \
      -DCMAKE_CXX_COMPILER=/rustroot/bin/g++ \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/rustroot \
      -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
      -DCOMPILER_RT_BUILD_XRAY=OFF \
      -DCOMPILER_RT_BUILD_MEMPROF=OFF \
      -DCOMPILER_RT_BUILD_CTX_PROFILE=OFF \
      -DLLVM_TARGETS_TO_BUILD=$LLVM_BUILD_TARGETS \
      -DLLVM_INCLUDE_BENCHMARKS=OFF \
      -DLLVM_INCLUDE_TESTS=OFF \
      -DLLVM_INCLUDE_EXAMPLES=OFF \
      -DLLVM_ENABLE_PROJECTS="clang;lld;bolt" \
      -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
      -DLLVM_BINUTILS_INCDIR="/rustroot/lib/gcc/$GCC_PLUGIN_TARGET/$GCC_VERSION/plugin/include/" \
      -DRUNTIMES_CMAKE_ARGS="-DCMAKE_CXX_FLAGS=\"--gcc-toolchain=/rustroot\"" \
      -DC_INCLUDE_DIRS="$INC"

hide_output make -j$(nproc)
hide_output make install

cd ../..
rm -rf llvm-project
