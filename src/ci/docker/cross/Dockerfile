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
  gcc-sparc64-linux-gnu \
  libc6-dev-sparc64-cross \
  bzip2 \
  patch \
  libssl-dev \
  pkg-config

WORKDIR /tmp

COPY cross/build-rumprun.sh /tmp/
RUN ./build-rumprun.sh

COPY cross/build-arm-musl.sh /tmp/
RUN ./build-arm-musl.sh

COPY cross/install-mips-musl.sh /tmp/
RUN ./install-mips-musl.sh

COPY cross/install-mipsel-musl.sh /tmp/
RUN ./install-mipsel-musl.sh

COPY cross/install-x86_64-redox.sh /tmp/
RUN ./install-x86_64-redox.sh

ENV TARGETS=asmjs-unknown-emscripten
ENV TARGETS=$TARGETS,wasm32-unknown-emscripten
ENV TARGETS=$TARGETS,x86_64-rumprun-netbsd
ENV TARGETS=$TARGETS,mips-unknown-linux-musl
ENV TARGETS=$TARGETS,mipsel-unknown-linux-musl
ENV TARGETS=$TARGETS,arm-unknown-linux-musleabi
ENV TARGETS=$TARGETS,arm-unknown-linux-musleabihf
ENV TARGETS=$TARGETS,armv7-unknown-linux-musleabihf
ENV TARGETS=$TARGETS,sparc64-unknown-linux-gnu
ENV TARGETS=$TARGETS,x86_64-unknown-redox

ENV CC_mipsel_unknown_linux_musl=mipsel-openwrt-linux-gcc \
    CC_mips_unknown_linux_musl=mips-openwrt-linux-gcc \
    CC_sparc64_unknown_linux_gnu=sparc64-linux-gnu-gcc \
    CC_x86_64_unknown_redox=x86_64-unknown-redox-gcc

# Suppress some warnings in the openwrt toolchains we downloaded
ENV STAGING_DIR=/tmp

ENV RUST_CONFIGURE_ARGS \
      --enable-extended \
      --target=$TARGETS \
      --musl-root-arm=/usr/local/arm-linux-musleabi \
      --musl-root-armhf=/usr/local/arm-linux-musleabihf \
      --musl-root-armv7=/usr/local/armv7-linux-musleabihf
ENV SCRIPT python2.7 ../x.py dist --target $TARGETS

# sccache
COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh
