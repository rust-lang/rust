FROM ubuntu:16.04

RUN dpkg --add-architecture i386 && \
    apt-get update && \
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
  expect \
  openjdk-9-jre \
  sudo \
  libstdc++6:i386 \
  xz-utils \
  libssl-dev \
  pkg-config

WORKDIR /android/
ENV PATH=$PATH:/android/ndk-arm-9/bin:/android/sdk/tools:/android/sdk/platform-tools

COPY install-ndk.sh install-sdk.sh accept-licenses.sh /android/
RUN sh /android/install-ndk.sh
RUN sh /android/install-sdk.sh

RUN curl -OL https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && \
    rm dumb-init_*.deb

COPY start-emulator.sh /android/

ENTRYPOINT ["/usr/bin/dumb-init", "--", "/android/start-emulator.sh"]

RUN curl -o /usr/local/bin/sccache \
      https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/2017-02-24-sccache-x86_64-unknown-linux-gnu && \
      chmod +x /usr/local/bin/sccache

ENV RUST_CONFIGURE_ARGS \
      --target=arm-linux-androideabi \
      --arm-linux-androideabi-ndk=/android/ndk-arm-9

ENV SCRIPT python2.7 ../x.py test --target arm-linux-androideabi
