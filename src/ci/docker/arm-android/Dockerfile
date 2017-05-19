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
    download_and_make_toolchain android-ndk-r13b-linux-x86_64.zip arm 9

# sdk
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
  libgl1-mesa-glx \
  libpulse0 \
  libstdc++6:i386 \
  openjdk-9-jre-headless \
  tzdata

COPY scripts/android-sdk.sh /scripts/
RUN . /scripts/android-sdk.sh && \
    download_and_create_avd tools_r25.2.5-linux.zip armeabi-v7a 18

# env
ENV PATH=$PATH:/android/sdk/tools
ENV PATH=$PATH:/android/sdk/platform-tools

ENV TARGETS=arm-linux-androideabi

ENV RUST_CONFIGURE_ARGS \
      --target=$TARGETS \
      --arm-linux-androideabi-ndk=/android/ndk/arm-9

ENV SCRIPT python2.7 ../x.py test --target $TARGETS

# sccache
COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

# init
COPY scripts/android-start-emulator.sh /scripts/
ENTRYPOINT ["/usr/bin/dumb-init", "--", "/scripts/android-start-emulator.sh"]
