FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  clang \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python3 \
  git \
  cmake \
  sudo \
  bzip2 \
  xz-utils \
  texinfo \
  wget \
  libssl-dev \
  pkg-config \
  && rm -rf /var/lib/apt/lists/*

COPY scripts/freebsd-toolchain.sh /tmp/
RUN /tmp/freebsd-toolchain.sh x86_64

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY scripts/cmake.sh /scripts/
RUN /scripts/cmake.sh

ENV \
    AR_x86_64_unknown_freebsd=x86_64-unknown-freebsd12-ar \
    CC_x86_64_unknown_freebsd=x86_64-unknown-freebsd12-clang \
    CXX_x86_64_unknown_freebsd=x86_64-unknown-freebsd12-clang++

ENV HOSTS=x86_64-unknown-freebsd

ENV RUST_CONFIGURE_ARGS \
    --enable-full-tools \
    --enable-extended \
    --enable-profiler \
    --enable-sanitizers \
    --disable-docs
ENV SCRIPT python3 ../x.py dist --host $HOSTS --target $HOSTS
