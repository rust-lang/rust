FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  file \
  curl \
  ca-certificates \
  python \
  git \
  cmake \
  sudo \
  gdb \
  xz-utils \
  wget \
  patch

RUN curl -sL https://nodejs.org/dist/v9.2.0/node-v9.2.0-linux-x64.tar.xz | \
  tar -xJ

WORKDIR /build/
COPY scripts/musl-toolchain.sh /build/
RUN bash musl-toolchain.sh x86_64 && rm -rf build
WORKDIR /

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_CONFIGURE_ARGS \
  --musl-root-x86_64=/usr/local/x86_64-linux-musl \
  --set build.nodejs=/node-v9.2.0-linux-x64/bin/node \
  --set rust.lld

# Some run-make tests have assertions about code size, and enabling debug
# assertions in libstd causes the binary to be much bigger than it would
# otherwise normally be. We already test libstd with debug assertions in lots of
# other contexts as well
ENV NO_DEBUG_ASSERTIONS=1

ENV WASM_TARGETS=wasm32-unknown-unknown
ENV WASM_SCRIPT python2.7 /checkout/x.py test --target $WASM_TARGETS \
  src/test/run-make \
  src/test/ui \
  src/test/run-pass \
  src/test/compile-fail \
  src/test/mir-opt \
  src/test/codegen-units \
  src/libcore

ENV NVPTX_TARGETS=nvptx64-nvidia-cuda
ENV NVPTX_SCRIPT python2.7 /checkout/x.py test --target $NVPTX_TARGETS \
  src/test/run-make \
  src/test/assembly

ENV MUSL_TARGETS=x86_64-unknown-linux-musl \
    CC_x86_64_unknown_linux_musl=x86_64-linux-musl-gcc \
    CXX_x86_64_unknown_linux_musl=x86_64-linux-musl-g++
ENV MUSL_SCRIPT python2.7 /checkout/x.py test --target $MUSL_TARGETS

ENV SCRIPT $WASM_SCRIPT && $NVPTX_SCRIPT && $MUSL_SCRIPT
