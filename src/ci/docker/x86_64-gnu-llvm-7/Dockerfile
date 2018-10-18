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
  libedit-dev \
  zlib1g-dev \
  xz-utils

RUN  curl https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
  echo deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-7 main > /etc/apt/sources.list.d/llvm.list && \
  apt-get update && apt-get install -y --no-install-recommends llvm-7-tools

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

# using llvm-link-shared due to libffi issues -- see #34486
ENV RUST_CONFIGURE_ARGS \
      --build=x86_64-unknown-linux-gnu \
      --llvm-root=/usr/lib/llvm-7 \
      --enable-llvm-link-shared
ENV RUST_CHECK_TARGET check
