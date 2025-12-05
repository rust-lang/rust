FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  autoconf \
  automake \
  bison \
  bzip2 \
  ca-certificates \
  cmake \
  curl \
  file \
  flex \
  g++ \
  gawk \
  git \
  libcurl4-openssl-dev \
  libssl-dev \
  make \
  ninja-build \
  nasm \
  pkg-config \
  python3 \
  sudo \
  texinfo \
  wget \
  xz-utils \
  zlib1g-dev

COPY host-x86_64/dist-x86_64-haiku/llvm-config.sh /bin/llvm-config-haiku

ENV ARCH=x86_64

WORKDIR /tmp
COPY host-x86_64/dist-x86_64-haiku/build-toolchain.sh /tmp/
RUN /tmp/build-toolchain.sh $ARCH

COPY host-x86_64/dist-x86_64-haiku/fetch-packages.sh /tmp/
RUN /tmp/fetch-packages.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV HOST=x86_64-unknown-haiku
ENV TARGET=target.$HOST

ENV RUST_CONFIGURE_ARGS --disable-jemalloc \
  --set=$TARGET.cc=x86_64-unknown-haiku-gcc \
  --set=$TARGET.cxx=x86_64-unknown-haiku-g++ \
  --set=$TARGET.llvm-config=/bin/llvm-config-haiku
ENV EXTERNAL_LLVM 1

ENV SCRIPT python3 ../x.py dist --host=$HOST --target=$HOST
