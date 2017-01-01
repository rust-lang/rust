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
  gcc-aarch64-linux-gnu libc6-dev-arm64-cross \
  gcc-arm-linux-gnueabi libc6-dev-armel-cross \
  gcc-arm-linux-gnueabihf libc6-dev-armhf-cross \
  gcc-mips-linux-gnu libc6-dev-mips-cross \
  gcc-mipsel-linux-gnu libc6-dev-mipsel-cross \
  gcc-mips64-linux-gnuabi64 libc6-dev-mips64-cross \
  gcc-mips64el-linux-gnuabi64 libc6-dev-mips64el-cross \
  gcc-powerpc-linux-gnu libc6-dev-powerpc-cross \
  gcc-powerpc64-linux-gnu libc6-dev-ppc64-cross \
  gcc-powerpc64le-linux-gnu libc6-dev-ppc64el-cross \
  gcc-s390x-linux-gnu libc6-dev-s390x-cross \
  xz-utils

ENV SCCACHE_DIGEST=7237e38e029342fa27b7ac25412cb9d52554008b12389727320bd533fd7f05b6a96d55485f305caf95e5c8f5f97c3313e10012ccad3e752aba2518f3522ba783
RUN curl -L https://api.pub.build.mozilla.org/tooltool/sha512/$SCCACHE_DIGEST | \
      tar xJf - -C /usr/local/bin --strip-components=1

RUN curl -OL https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && \
    rm dumb-init_*.deb
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

ENV TARGETS=aarch64-unknown-linux-gnu
ENV TARGETS=$TARGETS,arm-unknown-linux-gnueabi
ENV TARGETS=$TARGETS,arm-unknown-linux-gnueabihf
ENV TARGETS=$TARGETS,armv7-unknown-linux-gnueabihf
ENV TARGETS=$TARGETS,asmjs-unknown-emscripten
ENV TARGETS=$TARGETS,mips-unknown-linux-gnu
ENV TARGETS=$TARGETS,mips64-unknown-linux-gnuabi64
ENV TARGETS=$TARGETS,mips64el-unknown-linux-gnuabi64
ENV TARGETS=$TARGETS,mipsel-unknown-linux-gnu
ENV TARGETS=$TARGETS,powerpc-unknown-linux-gnu
ENV TARGETS=$TARGETS,powerpc64-unknown-linux-gnu
ENV TARGETS=$TARGETS,powerpc64le-unknown-linux-gnu
ENV TARGETS=$TARGETS,s390x-unknown-linux-gnu
ENV TARGETS=$TARGETS,wasm32-unknown-emscripten

#ENV TARGETS=$TARGETS,mips-unknown-linux-musl
#ENV TARGETS=$TARGETS,arm-unknown-linux-musleabi
#ENV TARGETS=$TARGETS,arm-unknown-linux-musleabihf
#ENV TARGETS=$TARGETS,armv7-unknown-linux-musleabihf
#ENV TARGETS=$TARGETS,x86_64-rumprun-netbsd

ENV RUST_CONFIGURE_ARGS \
      --target=$TARGETS \
      --enable-rustbuild

# Just a smoke test in dist to see if this works for now, we should expand this
# to all the targets above eventually.
ENV SCRIPT \
  python2.7 ../x.py build && \
  python2.7 ../x.py dist --target wasm32-unknown-emscripten

ENV AR_s390x_unknown_linux_gnu=s390x-linux-gnu-ar \
    CC_s390x_unknown_linux_gnu=s390x-linux-gnu-gcc \
    AR_mips64_unknown_linux_gnuabi64=mips64-linux-gnuabi64-ar \
    CC_mips64_unknown_linux_gnuabi64=mips64-linux-gnuabi64-gcc \
    AR_mips64el_unknown_linux_gnuabi64=mips64el-linux-gnuabi64-ar \
    CC_mips64el_unknown_linux_gnuabi64=mips64el-linux-gnuabi64-gcc \
    AR_powerpc64_unknown_linux_gnu=powerpc64-linux-gnu-ar \
    CC_powerpc64_unknown_linux_gnu=powerpc64-linux-gnu-gcc

# FIXME(rust-lang/rust#36150): powerpc unfortunately aborts right now
ENV NO_LLVM_ASSERTIONS=1
