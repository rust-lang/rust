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

RUN curl -o /usr/local/bin/sccache \
      https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/2017-02-24-sccache-x86_64-unknown-linux-gnu && \
      chmod +x /usr/local/bin/sccache

RUN curl -OL https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && \
    rm dumb-init_*.deb
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

WORKDIR /tmp

COPY build-rumprun.sh /tmp/
RUN ./build-rumprun.sh

COPY build-arm-musl.sh /tmp/
RUN ./build-arm-musl.sh

# originally from
# https://downloads.openwrt.org/snapshots/trunk/ar71xx/generic/OpenWrt-Toolchain-ar71xx-generic_gcc-5.3.0_musl-1.1.16.Linux-x86_64.tar.bz2
RUN mkdir /usr/local/mips-linux-musl
RUN curl -L https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/OpenWrt-Toolchain-ar71xx-generic_gcc-5.3.0_musl-1.1.16.Linux-x86_64.tar.bz2 | \
      tar xjf - -C /usr/local/mips-linux-musl --strip-components=2
RUN for file in /usr/local/mips-linux-musl/bin/mips-openwrt-linux-*; do \
      ln -s $file /usr/local/bin/`basename $file`; \
    done

# Note that this originally came from:
# https://downloads.openwrt.org/snapshots/trunk/malta/generic/OpenWrt-Toolchain-malta-le_gcc-5.3.0_musl-1.1.15.Linux-x86_64.tar.bz2
RUN mkdir /usr/local/mipsel-linux-musl
RUN curl -L https://s3.amazonaws.com/rust-lang-ci/libc/OpenWrt-Toolchain-malta-le_gcc-5.3.0_musl-1.1.15.Linux-x86_64.tar.bz2 | \
      tar xjf - -C /usr/local/mipsel-linux-musl --strip-components=2
RUN for file in /usr/local/mipsel-linux-musl/bin/mipsel-openwrt-linux-*; do \
      ln -s $file /usr/local/bin/`basename $file`; \
    done

ENV TARGETS=asmjs-unknown-emscripten
ENV TARGETS=$TARGETS,wasm32-unknown-emscripten
ENV TARGETS=$TARGETS,x86_64-rumprun-netbsd
ENV TARGETS=$TARGETS,mips-unknown-linux-musl
ENV TARGETS=$TARGETS,mipsel-unknown-linux-musl
ENV TARGETS=$TARGETS,arm-unknown-linux-musleabi
ENV TARGETS=$TARGETS,arm-unknown-linux-musleabihf
ENV TARGETS=$TARGETS,armv7-unknown-linux-musleabihf
ENV TARGETS=$TARGETS,sparc64-unknown-linux-gnu

ENV CC_mipsel_unknown_linux_musl=mipsel-openwrt-linux-gcc \
    CC_mips_unknown_linux_musl=mips-openwrt-linux-gcc \
    CC_sparc64_unknown_linux_gnu=sparc64-linux-gnu-gcc

# Suppress some warnings in the openwrt toolchains we downloaded
ENV STAGING_DIR=/tmp

ENV RUST_CONFIGURE_ARGS \
      --target=$TARGETS \
      --musl-root-arm=/usr/local/arm-linux-musleabi \
      --musl-root-armhf=/usr/local/arm-linux-musleabihf \
      --musl-root-armv7=/usr/local/armv7-linux-musleabihf
ENV SCRIPT python2.7 ../x.py dist --target $TARGETS
