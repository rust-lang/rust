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
  ccache \
  sudo \
  gdb \
  llvm-3.7-tools \
  libedit-dev \
  zlib1g-dev

ENV RUST_CONFIGURE_ARGS \
      --build=x86_64-unknown-linux-gnu \
      --llvm-root=/usr/lib/llvm-3.7
ENV RUST_CHECK_TARGET check
RUN mkdir /tmp/obj
RUN chmod 777 /tmp/obj
