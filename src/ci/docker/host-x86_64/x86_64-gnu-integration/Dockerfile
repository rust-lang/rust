FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  clang-15 \
  g++ \
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
# FIXME: Move to canonical triple
ENV \
    AR_x86_64_fuchsia=x86_64-unknown-fuchsia-ar \
    CC_x86_64_fuchsia=x86_64-unknown-fuchsia-clang \
    CFLAGS_x86_64_fuchsia="--target=x86_64-unknown-fuchsia --sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/sysroot -I/usr/local/core-linux-amd64-fuchsia-sdk/pkg/fdio/include" \
    CXX_x86_64_fuchsia=x86_64-unknown-fuchsia-clang++ \
    CXXFLAGS_x86_64_fuchsia="--target=x86_64-unknown-fuchsia --sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/sysroot -I/usr/local/core-linux-amd64-fuchsia-sdk/pkg/fdio/include" \
    LDFLAGS_x86_64_fuchsia="--target=x86_64-unknown-fuchsia --sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/sysroot -L/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/lib"

WORKDIR /tmp
COPY scripts/shared.sh /tmp/
COPY scripts/build-fuchsia-toolchain.sh /tmp/
RUN /tmp/build-fuchsia-toolchain.sh

ENV CARGO_TARGET_X86_64_FUCHSIA_AR /usr/local/bin/llvm-ar
ENV CARGO_TARGET_X86_64_FUCHSIA_RUSTFLAGS \
  -C panic=abort \
  -C force-unwind-tables=yes \
  -C link-arg=--sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/sysroot \
  -Lnative=/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/sysroot/lib \
  -Lnative=/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/lib

ENV TARGETS=x86_64-fuchsia
ENV TARGETS=$TARGETS,x86_64-unknown-linux-gnu
ENV TARGETS=$TARGETS,wasm32-unknown-unknown

# Fuchsia clang does not have wasm target enabled, use system clang.
ENV CC_wasm32_unknown_unknown=clang-15

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_INSTALL_DIR /checkout/obj/install
RUN mkdir -p $RUST_INSTALL_DIR/etc

# Fuchsia only supports LLVM.
ENV CODEGEN_BACKENDS llvm

ENV RUST_CONFIGURE_ARGS \
  --prefix=$RUST_INSTALL_DIR \
  --sysconfdir=etc \
  --enable-lld \
  --llvm-libunwind=in-tree \
  --enable-extended \
  --disable-docs \
  --set target.x86_64-fuchsia.cc=/usr/local/bin/clang \
  --set target.x86_64-fuchsia.cxx=/usr/local/bin/clang++ \
  --set target.x86_64-fuchsia.ar=/usr/local/bin/llvm-ar \
  --set target.x86_64-fuchsia.ranlib=/usr/local/bin/llvm-ranlib \
  --set target.x86_64-fuchsia.linker=/usr/local/bin/ld.lld
ENV SCRIPT \
    python3 ../x.py install --target $TARGETS compiler/rustc library/std clippy && \
    bash ../src/ci/docker/host-x86_64/x86_64-gnu-integration/build-fuchsia.sh
