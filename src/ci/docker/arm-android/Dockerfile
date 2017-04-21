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
        --arch arm \
        --api 9 && \
    rm -rf android-ndk-*

RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
  expect \
  openjdk-9-jre-headless \
  libstdc++6:i386

WORKDIR /android/
ENV PATH=$PATH:/android-ndk/bin
ENV PATH=$PATH:/android/sdk/tools
ENV PATH=$PATH:/android/sdk/platform-tools

COPY accept-licenses.sh /android/
COPY install-sdk.sh /android/
RUN sh /android/install-sdk.sh

COPY start-emulator.sh /android/
ENTRYPOINT ["/usr/bin/dumb-init", "--", "/android/start-emulator.sh"]

ENV RUST_CONFIGURE_ARGS \
      --target=arm-linux-androideabi \
      --arm-linux-androideabi-ndk=/android-ndk

ENV SCRIPT python2.7 ../x.py test --target arm-linux-androideabi
