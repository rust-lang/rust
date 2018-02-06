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
  zlib1g-dev \
  g++-arm-linux-gnueabi \
  g++-arm-linux-gnueabihf \
  g++-aarch64-linux-gnu \
  gcc-sparc64-linux-gnu \
  libc6-dev-sparc64-cross \
  bzip2 \
  patch \
  libssl-dev \
  pkg-config

WORKDIR /build

COPY dist-various-1/build-rumprun.sh /build
RUN ./build-rumprun.sh

COPY dist-various-1/install-x86_64-redox.sh /build
RUN ./install-x86_64-redox.sh

COPY scripts/musl.sh /build
RUN env \
    CC=arm-linux-gnueabi-gcc CFLAGS="-march=armv6 -marm" \
    CXX=arm-linux-gnueabi-g++ CXXFLAGS="-march=armv6 -marm" \
    bash musl.sh arm && \
    env \
    CC=arm-linux-gnueabihf-gcc CFLAGS="-march=armv6 -marm" \
    CXX=arm-linux-gnueabihf-g++ CXXFLAGS="-march=armv6 -marm" \
    bash musl.sh armhf && \
    env \
    CC=arm-linux-gnueabihf-gcc CFLAGS="-march=armv7-a" \
    CXX=arm-linux-gnueabihf-g++ CXXFLAGS="-march=armv7-a" \
    bash musl.sh armv7 && \
    env \
    CC=aarch64-linux-gnu-gcc \
    CXX=aarch64-linux-gnu-g++ \
    bash musl.sh aarch64 && \
    rm -rf /build/*

COPY dist-various-1/install-mips-musl.sh /build
RUN ./install-mips-musl.sh

COPY dist-various-1/install-mipsel-musl.sh /build
RUN ./install-mipsel-musl.sh

ENV TARGETS=asmjs-unknown-emscripten
ENV TARGETS=$TARGETS,wasm32-unknown-emscripten
ENV TARGETS=$TARGETS,x86_64-rumprun-netbsd
ENV TARGETS=$TARGETS,mips-unknown-linux-musl
ENV TARGETS=$TARGETS,mipsel-unknown-linux-musl
ENV TARGETS=$TARGETS,arm-unknown-linux-musleabi
ENV TARGETS=$TARGETS,arm-unknown-linux-musleabihf
ENV TARGETS=$TARGETS,armv5te-unknown-linux-gnueabi
ENV TARGETS=$TARGETS,armv7-unknown-linux-musleabihf
ENV TARGETS=$TARGETS,aarch64-unknown-linux-musl
ENV TARGETS=$TARGETS,sparc64-unknown-linux-gnu
ENV TARGETS=$TARGETS,x86_64-unknown-redox

# FIXME: remove armv5te vars after https://github.com/alexcrichton/cc-rs/issues/271
#        get fixed and cc update
ENV CC_mipsel_unknown_linux_musl=mipsel-openwrt-linux-gcc \
    CC_mips_unknown_linux_musl=mips-openwrt-linux-gcc \
    CC_sparc64_unknown_linux_gnu=sparc64-linux-gnu-gcc \
    CC_x86_64_unknown_redox=x86_64-unknown-redox-gcc \
    CC_armv5te_unknown_linux_gnueabi=arm-linux-gnueabi-gcc \
    CFLAGS_armv5te_unknown_linux_gnueabi="-march=armv5te -marm -mfloat-abi=soft"

# Suppress some warnings in the openwrt toolchains we downloaded
ENV STAGING_DIR=/tmp

ENV RUST_CONFIGURE_ARGS \
      --enable-extended \
      --target=$TARGETS \
      --musl-root-arm=/musl-arm \
      --musl-root-armhf=/musl-armhf \
      --musl-root-armv7=/musl-armv7 \
      --musl-root-aarch64=/musl-aarch64
ENV SCRIPT python2.7 ../x.py dist --target $TARGETS

# sccache
COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh
