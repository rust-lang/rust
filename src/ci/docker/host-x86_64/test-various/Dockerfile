FROM ubuntu:20.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  clang-11 \
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
  xz-utils \
  wget \
  patch \
  ovmf \
  qemu-efi-aarch64 \
  qemu-system-arm \
  qemu-system-x86 \
  && rm -rf /var/lib/apt/lists/*

RUN curl -sL https://nodejs.org/dist/v18.12.0/node-v18.12.0-linux-x64.tar.xz | \
  tar -xJ

# Install 32-bit OVMF files for the i686-unknown-uefi test. This package
# is not available in ubuntu 20.04, so download a 22.04 package.
RUN curl -sL --output ovmf-ia32.deb http://mirrors.kernel.org/ubuntu/pool/universe/e/edk2/ovmf-ia32_2022.02-3_all.deb
RUN dpkg -i ovmf-ia32.deb && rm ovmf-ia32.deb

WORKDIR /build/
COPY scripts/musl-toolchain.sh /build/
RUN bash musl-toolchain.sh x86_64 && rm -rf build
WORKDIR /

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_CONFIGURE_ARGS \
  --musl-root-x86_64=/usr/local/x86_64-linux-musl \
  --set build.nodejs=/node-v18.12.0-linux-x64/bin/node \
  --set rust.lld

# Some run-make tests have assertions about code size, and enabling debug
# assertions in libstd causes the binary to be much bigger than it would
# otherwise normally be. We already test libstd with debug assertions in lots of
# other contexts as well
ENV NO_DEBUG_ASSERTIONS=1
ENV NO_OVERFLOW_CHECKS=1

ENV WASM_TARGETS=wasm32-unknown-unknown
ENV WASM_SCRIPT python3 /checkout/x.py --stage 2 test --host='' --target $WASM_TARGETS \
  tests/run-make \
  tests/ui \
  tests/mir-opt \
  tests/codegen-units \
  tests/codegen \
  tests/assembly \
  library/core

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
