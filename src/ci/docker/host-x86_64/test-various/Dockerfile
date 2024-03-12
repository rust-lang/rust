FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  clang-11 \
  llvm-11 \
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

RUN curl -sL https://nodejs.org/dist/v18.12.0/node-v18.12.0-linux-x64.tar.xz | \
  tar -xJ

WORKDIR /build/
COPY scripts/musl-toolchain.sh /build/
RUN bash musl-toolchain.sh x86_64 && rm -rf build
WORKDIR /

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY host-x86_64/dist-various-2/build-wasi-toolchain.sh /tmp/
RUN /tmp/build-wasi-toolchain.sh

ENV RUST_CONFIGURE_ARGS \
  --musl-root-x86_64=/usr/local/x86_64-linux-musl \
  --set build.nodejs=/node-v18.12.0-linux-x64/bin/node \
  --set rust.lld \
  --set target.wasm32-wasip1.wasi-root=/wasm32-wasip1

# Some run-make tests have assertions about code size, and enabling debug
# assertions in libstd causes the binary to be much bigger than it would
# otherwise normally be. We already test libstd with debug assertions in lots of
# other contexts as well
ENV NO_DEBUG_ASSERTIONS=1
ENV NO_OVERFLOW_CHECKS=1

RUN curl -L https://github.com/bytecodealliance/wasmtime/releases/download/v18.0.2/wasmtime-v18.0.2-x86_64-linux.tar.xz | \
  tar -xJ
ENV PATH "$PATH:/wasmtime-v18.0.2-x86_64-linux"

ENV WASM_TARGETS=wasm32-wasip1
ENV WASM_SCRIPT python3 /checkout/x.py --stage 2 test --host='' --target $WASM_TARGETS \
  tests/run-make \
  tests/ui \
  tests/mir-opt \
  tests/codegen-units \
  tests/codegen \
  tests/assembly \
  library/core
ENV CC_wasm32_wasip1=clang-11 \
    CFLAGS_wasm32_wasip1="--sysroot /wasm32-wasip1" \
    AR_wasm32_wasip1=llvm-ar-11

ENV NVPTX_TARGETS=nvptx64-nvidia-cuda
ENV NVPTX_SCRIPT python3 /checkout/x.py --stage 2 test --host='' --target $NVPTX_TARGETS \
  tests/run-make \
  tests/assembly

ENV MUSL_TARGETS=x86_64-unknown-linux-musl \
    CC_x86_64_unknown_linux_musl=x86_64-linux-musl-gcc \
    CXX_x86_64_unknown_linux_musl=x86_64-linux-musl-g++
ENV MUSL_SCRIPT python3 /checkout/x.py --stage 2 test --host='' --target $MUSL_TARGETS

COPY host-x86_64/test-various/uefi_qemu_test /uefi_qemu_test
ENV UEFI_TARGETS=aarch64-unknown-uefi,i686-unknown-uefi,x86_64-unknown-uefi \
    CC_aarch64_unknown_uefi=clang-11 \
    CXX_aarch64_unknown_uefi=clang++-11 \
    CC_i686_unknown_uefi=clang-11 \
    CXX_i686_unknown_uefi=clang++-11 \
    CC_x86_64_unknown_uefi=clang-11 \
    CXX_x86_64_unknown_uefi=clang++-11
ENV UEFI_SCRIPT python3 /checkout/x.py --stage 2 build --host='' --target $UEFI_TARGETS && \
  python3 -u /uefi_qemu_test/run.py

ENV SCRIPT $WASM_SCRIPT && $NVPTX_SCRIPT && $MUSL_SCRIPT && $UEFI_SCRIPT
