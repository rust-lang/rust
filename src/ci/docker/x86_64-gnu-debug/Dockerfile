FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  file \
  curl \
  ca-certificates \
  python2.7 \
  python2.7-dev \
  libxml2-dev \
  libncurses-dev \
  libedit-dev \
  swig \
  doxygen \
  git \
  cmake \
  sudo \
  gdb \
  xz-utils

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUSTBUILD_FORCE_CLANG_BASED_TESTS 1
ENV RUN_CHECK_WITH_PARALLEL_QUERIES 1

ENV RUST_CONFIGURE_ARGS \
      --build=x86_64-unknown-linux-gnu \
      --enable-debug \
      --enable-lld \
      --enable-lldb \
      --enable-optimize
ENV SCRIPT \
  python2.7 ../x.py build && \
  python2.7 ../x.py test src/test/run-make-fulldeps --test-args clang
