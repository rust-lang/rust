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

# GNU Arm Embedded Toolchain 7-2018-q2-update (June 27,2018)
ENV BASE_URL=https://developer.arm.com/-/media/Files/downloads/gnu-rm/
RUN curl -L $BASE_URL/7-2018q2/gcc-arm-none-eabi-7-2018-q2-update-linux.tar.bz2 | tar -xj
ENV PATH=$PATH:/gcc-arm-none-eabi-7-2018-q2-update/bin

ENV TARGET=armv7r-none-eabihf

ENV CC_armv7r_none_eabihf=arm-none-eabi-gcc \
    CFLAGS_armv7r_none_eabihf="-march=armv7-r"

ENV RUST_CONFIGURE_ARGS --disable-docs

ENV SCRIPT python2.7 ../x.py dist --target $TARGET
