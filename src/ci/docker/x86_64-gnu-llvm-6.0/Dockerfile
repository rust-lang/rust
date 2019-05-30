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
  llvm-6.0-tools \
  libedit-dev \
  zlib1g-dev \
  xz-utils

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

# using llvm-link-shared due to libffi issues -- see #34486
ENV RUST_CONFIGURE_ARGS \
      --build=x86_64-unknown-linux-gnu \
      --llvm-root=/usr/lib/llvm-6.0 \
      --enable-llvm-link-shared
ENV SCRIPT python2.7 ../x.py test src/tools/tidy && python2.7 ../x.py test
