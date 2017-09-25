FROM ubuntu:16.04

RUN apt-get update && apt-get build-dep -y clang llvm && apt-get install -y \
  build-essential \
  bzip2 \
  ca-certificates \
  cmake \
  curl \
  file \
  g++ \
  gdb \
  git \
  libedit-dev \
  make \
  ninja-build \
  nodejs \
  python2.7-dev \
  sudo \
  xz-utils \
  unzip

WORKDIR /tmp
COPY dist-fuchsia/shared.sh dist-fuchsia/build-toolchain.sh /tmp/
RUN /tmp/build-toolchain.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV \
    AR_x86_64_unknown_fuchsia=x86_64-unknown-fuchsia-ar \
    CC_x86_64_unknown_fuchsia=x86_64-unknown-fuchsia-clang \
    CXX_x86_64_unknown_fuchsia=x86_64-unknown-fuchsia-clang++ \
    AR_aarch64_unknown_fuchsia=aarch64-unknown-fuchsia-ar \
    CC_aarch64_unknown_fuchsia=aarch64-unknown-fuchsia-clang \
    CXX_aarch64_unknown_fuchsia=aarch64-unknown-fuchsia-clang++

ENV TARGETS=x86_64-unknown-fuchsia
ENV TARGETS=$TARGETS,aarch64-unknown-fuchsia

ENV RUST_CONFIGURE_ARGS --target=$TARGETS --enable-extended
ENV SCRIPT python2.7 ../x.py dist --target $TARGETS