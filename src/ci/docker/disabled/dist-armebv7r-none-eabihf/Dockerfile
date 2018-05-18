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
  xz-utils \
  bzip2 \
  libssl-dev \
  pkg-config


COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV BASE_URL=https://releases.linaro.org/components/toolchain/binaries/latest/armeb-eabi/
ENV GCC_LINARO=gcc-linaro-7.2.1-2017.11-x86_64_armeb-eabi

RUN curl -sL $BASE_URL/$GCC_LINARO.tar.xz | tar -xJ

ENV PATH=$PATH:/$GCC_LINARO/bin

ENV TARGET=armebv7r-none-eabihf

ENV CC_armebv7r_none_eabihf=armeb-eabi-gcc \
    CFLAGS_armebv7r_none_eabihf="-march=armv7-r"

ENV RUST_CONFIGURE_ARGS --disable-docs

ENV SCRIPT python2.7 ../x.py dist --target $TARGET
