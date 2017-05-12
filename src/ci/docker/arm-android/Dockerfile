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
      https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/2017-05-12-sccache-x86_64-unknown-linux-musl && \
    chmod +x /usr/local/bin/sccache

# Install NDK
COPY install-ndk.sh /tmp
RUN . /tmp/install-ndk.sh && \
    download_ndk android-ndk-r13b-linux-x86_64.zip && \
    make_standalone_toolchain arm 9 && \
    remove_ndk

# Install SDK
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
  openjdk-9-jre-headless \
  tzdata \
  libstdc++6:i386 \
  libgl1-mesa-glx \
  libpulse0

COPY install-sdk.sh /tmp
RUN . /tmp/install-sdk.sh && \
    download_sdk tools_r25.2.5-linux.zip && \
    download_sysimage armeabi-v7a 18 && \
    create_avd armeabi-v7a 18

# Setup env
ENV PATH=$PATH:/android/sdk/tools
ENV PATH=$PATH:/android/sdk/platform-tools

ENV TARGETS=arm-linux-androideabi

ENV RUST_CONFIGURE_ARGS \
      --target=$TARGETS \
      --arm-linux-androideabi-ndk=/android/ndk/arm-9

ENV SCRIPT python2.7 ../x.py test --target $TARGETS --verbose

# Entrypoint
COPY start-emulator.sh /android/
ENTRYPOINT ["/usr/bin/dumb-init", "--", "/android/start-emulator.sh"]
