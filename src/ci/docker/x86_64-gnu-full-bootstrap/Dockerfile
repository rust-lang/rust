FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  file \
  curl \
  ca-certificates \
  python2.7 \
  git \
  cmake \
  sudo \
  gdb \
  xz-utils

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_CONFIGURE_ARGS \
      --build=x86_64-unknown-linux-gnu \
      --enable-full-bootstrap
ENV SCRIPT python2.7 ../x.py build

# In general this just slows down the build and we're just a smoke test that
# a full bootstrap works in general, so there's not much need to take this
# penalty in build times.
ENV NO_LLVM_ASSERTIONS 1
ENV NO_DEBUG_ASSERTIONS 1
