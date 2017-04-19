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
      https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/2017-04-04-sccache-x86_64-unknown-linux-musl && \
      chmod +x /usr/local/bin/sccache

RUN curl -O https://dl.google.com/android/repository/android-ndk-r13b-linux-x86_64.zip && \
    unzip -q android-ndk-r13b-linux-x86_64.zip && \
    python2.7 android-ndk-r13b/build/tools/make_standalone_toolchain.py \
        --install-dir /android-ndk \
        --arch arm64 \
        --api 21 && \
    rm -rf android-ndk-*

ENTRYPOINT ["/usr/bin/dumb-init", "--"]

ENV PATH=$PATH:/android-ndk/bin

ENV DEP_Z_ROOT=/android-ndk/sysroot/usr/

ENV RUST_CONFIGURE_ARGS \
      --target=aarch64-linux-android \
      --host=aarch64-linux-android \
      --aarch64-linux-android-ndk=/android-ndk \
      --disable-rpath \
      --enable-extended \
      --enable-cargo-openssl-static

ENV SCRIPT python2.7 ../x.py dist --target aarch64-linux-android --host aarch64-linux-android
