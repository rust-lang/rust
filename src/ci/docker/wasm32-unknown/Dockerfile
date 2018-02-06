FROM ubuntu:16.04

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
  xz-utils

RUN curl -sL https://nodejs.org/dist/v9.2.0/node-v9.2.0-linux-x64.tar.xz | \
    tar -xJ

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV TARGETS=wasm32-unknown-unknown

ENV RUST_CONFIGURE_ARGS \
  --target=$TARGETS \
  --set build.nodejs=/node-v9.2.0-linux-x64/bin/node

ENV SCRIPT python2.7 /checkout/x.py test --target $TARGETS \
  src/test/ui \
  src/test/run-pass \
  src/test/compile-fail \
  src/test/parse-fail \
  src/test/mir-opt \
  src/test/codegen-units \
  src/libcore \
  src/libstd_unicode/ \
