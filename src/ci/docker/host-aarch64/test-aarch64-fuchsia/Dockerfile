# This job builds a toolchain capable of building Fuchsia, and then builds
# Fuchsia as an integration test of the toolchain. See the build-fuchsia.sh
# script in this directory for more details.

FROM ubuntu:26.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  clang \
  g++ \
  g++-x86-64-linux-gnu \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python3 \
  git \
  cmake \
  libssl-dev \
  sudo \
  xz-utils \
  pkg-config \
  unzip \
  && rm -rf /var/lib/apt/lists/*

# Duplicated in dist-various-2 Dockerfile.
ENV \
    AR_aarch64_unknown_fuchsia=aarch64-unknown-fuchsia-ar \
    CC_aarch64_unknown_fuchsia=aarch64-unknown-fuchsia-clang \
    CFLAGS_aarch64_unknown_fuchsia="\
        --target=aarch64-unknown-fuchsia \
        --sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/arm64/sysroot \
        -I/usr/local/core-linux-amd64-fuchsia-sdk/pkg/fdio/include" \
    CXX_aarch64_unknown_fuchsia=aarch64-unknown-fuchsia-clang++ \
    CXXFLAGS_aarch64_unknown_fuchsia="\
        --target=aarch64-unknown-fuchsia \
        --sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/arm64/sysroot \
        -I/usr/local/core-linux-amd64-fuchsia-sdk/pkg/fdio/include" \
    LDFLAGS_aarch64_unknown_fuchsia="\
        --target=aarch64-unknown-fuchsia \
        --sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/arm64/sysroot \
        -L/usr/local/core-linux-amd64-fuchsia-sdk/arch/arm64/lib" \
    AR_riscv64gc_unknown_fuchsia=riscv64gc-unknown-fuchsia-ar \
    CC_riscv64gc_unknown_fuchsia=riscv64gc-unknown-fuchsia-clang \
    CFLAGS_riscv64gc_unknown_fuchsia="\
        --target=riscv64-unknown-fuchsia \
        --sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/riscv64/sysroot \
        -I/usr/local/core-linux-amd64-fuchsia-sdk/pkg/fdio/include" \
    CXX_riscv64gc_unknown_fuchsia=riscv64gc-unknown-fuchsia-clang++ \
    CXXFLAGS_riscv64gc_unknown_fuchsia="\
        --target=riscv64-unknown-fuchsia \
        --sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/riscv64/sysroot \
        -I/usr/local/core-linux-amd64-fuchsia-sdk/pkg/fdio/include" \
    LDFLAGS_riscv64gc_unknown_fuchsia="\
        --target=riscv64-unknown-fuchsia \
        --sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/riscv64/sysroot \
        -L/usr/local/core-linux-amd64-fuchsia-sdk/arch/riscv64/lib" \
    AR_x86_64_unknown_fuchsia=x86_64-unknown-fuchsia-ar \
    CC_x86_64_unknown_fuchsia=x86_64-unknown-fuchsia-clang \
    CFLAGS_x86_64_unknown_fuchsia="\
        --target=x86_64-unknown-fuchsia \
        --sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/sysroot \
        -I/usr/local/core-linux-amd64-fuchsia-sdk/pkg/fdio/include" \
    CXX_x86_64_unknown_fuchsia=x86_64-unknown-fuchsia-clang++ \
    CXXFLAGS_x86_64_unknown_fuchsia="\
        --target=x86_64-unknown-fuchsia \
        --sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/sysroot \
        -I/usr/local/core-linux-amd64-fuchsia-sdk/pkg/fdio/include" \
    LDFLAGS_x86_64_unknown_fuchsia="\
        --target=x86_64-unknown-fuchsia \
        --sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/sysroot \
        -L/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/lib"

WORKDIR /tmp
COPY scripts/shared.sh /tmp/
COPY scripts/build-fuchsia-toolchain.sh /tmp/
RUN /tmp/build-fuchsia-toolchain.sh

ENV \
    CARGO_TARGET_AARCH64_UNKNOWN_FUCHSIA_AR="/usr/local/bin/llvm-ar" \
    CARGO_TARGET_AARCH64_UNKNOWN_FUCHSIA_RUSTFLAGS="-C panic=abort \
       -C force-unwind-tables=yes \
       -C link-arg=--sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/arm64/sysroot \
       -Lnative=/usr/local/core-linux-amd64-fuchsia-sdk/arch/arm64/sysroot/lib \
       -Lnative=/usr/local/core-linux-amd64-fuchsia-sdk/arch/arm64/lib" \
    CARGO_TARGET_RISCV64GC_UNKNOWN_FUCHSIA_AR="/usr/local/bin/llvm-ar" \
    CARGO_TARGET_RISCV64GC_UNKNOWN_FUCHSIA_RUSTFLAGS="-C panic=abort \
       -C force-unwind-tables=yes \
       -C link-arg=--sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/riscv64/sysroot \
       -Lnative=/usr/local/core-linux-amd64-fuchsia-sdk/arch/riscv64/sysroot/lib \
       -Lnative=/usr/local/core-linux-amd64-fuchsia-sdk/arch/riscv64/lib" \
    CARGO_TARGET_X86_64_UNKNOWN_FUCHSIA_AR="/usr/local/bin/llvm-ar" \
    CARGO_TARGET_X86_64_UNKNOWN_FUCHSIA_RUSTFLAGS="-C panic=abort \
       -C force-unwind-tables=yes \
       -C link-arg=--sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/sysroot \
       -Lnative=/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/sysroot/lib \
       -Lnative=/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/lib"

ENV TARGETS=aarch64-unknown-linux-gnu,x86_64-unknown-linux-gnu,aarch64-unknown-fuchsia,riscv64gc-unknown-fuchsia,x86_64-unknown-fuchsia

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_INSTALL_DIR="/checkout/obj/install"
RUN mkdir -p $RUST_INSTALL_DIR/etc

# Fuchsia only supports LLVM.
ENV CODEGEN_BACKENDS="llvm"

# download-rustc is not allowed for `x install`
ENV NO_DOWNLOAD_CI_RUSTC="1"

ENV RUST_CONFIGURE_ARGS="\
    --prefix=$RUST_INSTALL_DIR \
    --sysconfdir=etc \
    --enable-lld \
    --llvm-libunwind=in-tree \
    --enable-extended \
    --disable-docs \
    --set target.aarch64-unknown-fuchsia.cc=/usr/local/bin/clang \
    --set target.aarch64-unknown-fuchsia.cxx=/usr/local/bin/clang++ \
    --set target.aarch64-unknown-fuchsia.ar=/usr/local/bin/llvm-ar \
    --set target.aarch64-unknown-fuchsia.ranlib=/usr/local/bin/llvm-ranlib \
    --set target.aarch64-unknown-fuchsia.linker=/usr/local/bin/ld.lld \
    --set target.riscv64gc-unknown-fuchsia.cc=/usr/local/bin/clang \
    --set target.riscv64gc-unknown-fuchsia.cxx=/usr/local/bin/clang++ \
    --set target.riscv64gc-unknown-fuchsia.ar=/usr/local/bin/llvm-ar \
    --set target.riscv64gc-unknown-fuchsia.ranlib=/usr/local/bin/llvm-ranlib \
    --set target.riscv64gc-unknown-fuchsia.linker=/usr/local/bin/ld.lld \
    --set target.x86_64-unknown-fuchsia.cc=/usr/local/bin/clang \
    --set target.x86_64-unknown-fuchsia.cxx=/usr/local/bin/clang++ \
    --set target.x86_64-unknown-fuchsia.ar=/usr/local/bin/llvm-ar \
    --set target.x86_64-unknown-fuchsia.ranlib=/usr/local/bin/llvm-ranlib \
    --set target.x86_64-unknown-fuchsia.linker=/usr/local/bin/ld.lld"

ENV SCRIPT="\
    bash ../src/ci/docker/scripts/install-rust-for-fuchsia.sh && \
    bash ../src/ci/docker/scripts/build-fuchsia.sh"
