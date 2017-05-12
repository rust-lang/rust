FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
  g++ \
  make \
  file \
  curl \
  ca-certificates \
  python2.7 \
  git \
  cmake \
  unzip \
  sudo \
  xz-utils \
  libssl-dev \
  pkg-config

RUN curl -OL https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && \
    rm dumb-init_*.deb

RUN curl -o /usr/local/bin/sccache \
      https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/2017-04-29-sccache-x86_64-unknown-linux-musl && \
    chmod +x /usr/local/bin/sccache

ENTRYPOINT ["/usr/bin/dumb-init", "--"]

# Install NDK
COPY install-ndk.sh /tmp
RUN . /tmp/install-ndk.sh && \
    download_ndk android-ndk-r13b-linux-x86_64.zip && \
    make_standalone_toolchain arm 9 && \
    make_standalone_toolchain x86 9 && \
    make_standalone_toolchain arm64 21 && \
    make_standalone_toolchain x86_64 21 && \
    remove_ndk

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
