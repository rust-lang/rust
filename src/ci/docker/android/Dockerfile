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

ENV TARGETS=arm-linux-androideabi
ENV TARGETS=$TARGETS,i686-linux-android
ENV TARGETS=$TARGETS,aarch64-linux-android
ENV TARGETS=$TARGETS,armv7-linux-androideabi

ENV RUST_CONFIGURE_ARGS \
      --target=$TARGETS \
      --arm-linux-androideabi-ndk=/android/ndk-arm-9 \
      --armv7-linux-androideabi-ndk=/android/ndk-arm-9 \
      --i686-linux-android-ndk=/android/ndk-x86-9 \
      --aarch64-linux-android-ndk=/android/ndk-aarch64

# Just a smoke test in dist to see if this works for now, we should expand this
# to all the targets above eventually.
ENV SCRIPT \
  python2.7 ../x.py test --target arm-linux-androideabi && \
  python2.7 ../x.py dist --target $TARGETS
