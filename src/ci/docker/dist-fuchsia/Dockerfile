FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python2.7-dev \
  git \
  sudo \
  bzip2 \
  xz-utils \
  swig \
  libedit-dev \
  libncurses5-dev \
  patch \
  unzip

RUN curl -L https://cmake.org/files/v3.9/cmake-3.9.2-Linux-x86_64.tar.gz | \
      tar xzf - -C /usr/local --strip-components=1

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
