FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
  ca-certificates \
  cmake \
  curl \
  file \
  g++ \
  git \
  libssl-dev \
  make \
  pkg-config \
  python2.7 \
  sudo \
  unzip \
  xz-utils

# dumb-init
COPY scripts/dumb-init.sh /scripts/
RUN sh /scripts/dumb-init.sh

# ndk
COPY scripts/android-ndk.sh /scripts/
RUN . /scripts/android-ndk.sh && \
    download_and_make_toolchain android-ndk-r13b-linux-x86_64.zip x86_64 21

# env
ENV PATH=$PATH:/android/ndk/x86_64-21/bin

ENV DEP_Z_ROOT=/android/ndk/x86_64-21/sysroot/usr/

ENV HOSTS=x86_64-linux-android

ENV RUST_CONFIGURE_ARGS \
      --host=$HOSTS \
      --target=$HOSTS \
      --x86_64-linux-android-ndk=/android/ndk/x86_64-21 \
      --disable-rpath \
      --enable-extended \
      --enable-cargo-openssl-static

ENV SCRIPT python2.7 ../x.py dist --target $HOSTS --host $HOSTS

# sccache
COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

# init
ENTRYPOINT ["/usr/bin/dumb-init", "--"]
