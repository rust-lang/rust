FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  clang-11 \
  llvm-11 \
  gcc-multilib \
  g++ \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python3 \
  git \
  cmake \
  sudo \
  gdb \
  libssl-dev \
  pkg-config \
  bzip2 \
  xz-utils \
  wget \
  patch \
  ovmf \
  ovmf-ia32 \
  qemu-efi-aarch64 \
  qemu-system-arm \
  qemu-system-x86 \
  && rm -rf /var/lib/apt/lists/*

COPY scripts/nodejs.sh /scripts/
RUN sh /scripts/nodejs.sh /node
ENV PATH="/node/bin:${PATH}"

WORKDIR /build/
COPY scripts/musl-toolchain.sh /build/
RUN bash musl-toolchain.sh x86_64 && rm -rf build
WORKDIR /

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

RUN curl -L https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-27/wasi-sdk-27.0-x86_64-linux.tar.gz | \
  tar -xz
ENV WASI_SDK_PATH=/wasi-sdk-27.0-x86_64-linux

ENV RUST_CONFIGURE_ARGS \
  --musl-root-x86_64=/usr/local/x86_64-linux-musl \
  --set rust.lld

# Some run-make tests have assertions about code size, and enabling debug
# assertions in libstd causes the binary to be much bigger than it would
# otherwise normally be. We already test libstd with debug assertions in lots of
# other contexts as well
ENV NO_DEBUG_ASSERTIONS=1
ENV NO_OVERFLOW_CHECKS=1

RUN curl -L https://github.com/bytecodealliance/wasmtime/releases/download/v19.0.0/wasmtime-v19.0.0-x86_64-linux.tar.xz | \
  tar -xJ
ENV PATH "$PATH:/wasmtime-v19.0.0-x86_64-linux"

ENV WASM_WASIP_TARGET=wasm32-wasip1
ENV WASM_WASIP_SCRIPT python3 /checkout/x.py --stage 2 test --host='' --target $WASM_WASIP_TARGET \
  tests/run-make \
  tests/run-make-cargo \
  tests/ui \
  tests/mir-opt \
  tests/codegen-units \
  tests/codegen-llvm \
  tests/assembly-llvm \
  library/core

ENV NVPTX_TARGETS=nvptx64-nvidia-cuda
ENV NVPTX_SCRIPT python3 /checkout/x.py --stage 2 test --host='' --target $NVPTX_TARGETS \
  tests/run-make \
  tests/run-make-cargo \
  tests/assembly-llvm

ENV MUSL_TARGETS=x86_64-unknown-linux-musl \
    CC_x86_64_unknown_linux_musl=x86_64-linux-musl-gcc \
    CXX_x86_64_unknown_linux_musl=x86_64-linux-musl-g++
ENV MUSL_SCRIPT python3 /checkout/x.py --stage 2 test --host='' --target $MUSL_TARGETS

ENV UEFI_TARGETS=aarch64-unknown-uefi,i686-unknown-uefi,x86_64-unknown-uefi \
    CC_aarch64_unknown_uefi=clang-11 \
    CXX_aarch64_unknown_uefi=clang++-11 \
    CC_i686_unknown_uefi=clang-11 \
    CXX_i686_unknown_uefi=clang++-11 \
    CC_x86_64_unknown_uefi=clang-11 \
    CXX_x86_64_unknown_uefi=clang++-11
ENV UEFI_SCRIPT python3 /checkout/x.py --stage 2 build --host='' --target $UEFI_TARGETS && \
  python3 /checkout/x.py --stage 2 test tests/run-make-cargo/uefi-qemu/rmake.rs --target aarch64-unknown-uefi && \
  python3 /checkout/x.py --stage 2 test tests/run-make-cargo/uefi-qemu/rmake.rs --target i686-unknown-uefi && \
  python3 /checkout/x.py --stage 2 test tests/run-make-cargo/uefi-qemu/rmake.rs --target x86_64-unknown-uefi

ENV SCRIPT $WASM_WASIP_SCRIPT && $NVPTX_SCRIPT && $MUSL_SCRIPT && $UEFI_SCRIPT
