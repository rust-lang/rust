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
    download_ndk android-ndk-r13b-linux-x86_64.zip && \
    make_standalone_toolchain arm 9 && \
    make_standalone_toolchain x86 9 && \
    make_standalone_toolchain arm64 21 && \
    make_standalone_toolchain x86_64 21 && \
    remove_ndk

# env
ENV TARGETS=arm-linux-androideabi
ENV TARGETS=$TARGETS,armv7-linux-androideabi
ENV TARGETS=$TARGETS,i686-linux-android
ENV TARGETS=$TARGETS,aarch64-linux-android
ENV TARGETS=$TARGETS,x86_64-linux-android

ENV RUST_CONFIGURE_ARGS \
      --target=$TARGETS \
      --enable-extended \
      --arm-linux-androideabi-ndk=/android/ndk/arm-9 \
      --armv7-linux-androideabi-ndk=/android/ndk/arm-9 \
      --i686-linux-android-ndk=/android/ndk/x86-9 \
      --aarch64-linux-android-ndk=/android/ndk/arm64-21 \
      --x86_64-linux-android-ndk=/android/ndk/x86_64-21

ENV SCRIPT python2.7 ../x.py dist --target $TARGETS

# cache
COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

# init
ENTRYPOINT ["/usr/bin/dumb-init", "--"]
