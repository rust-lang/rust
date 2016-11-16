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
  ccache \
  unzip \
  expect \
  openjdk-9-jre \
  sudo \
  libstdc++6:i386

WORKDIR /android/
ENV PATH=$PATH:/android/ndk-arm-9/bin:/android/sdk/tools:/android/sdk/platform-tools

COPY install-ndk.sh install-sdk.sh accept-licenses.sh /android/
RUN sh /android/install-ndk.sh
RUN sh /android/install-sdk.sh

COPY start-emulator.sh /android/
ENTRYPOINT ["/android/start-emulator.sh"]

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
ENV XPY_CHECK test --target arm-linux-androideabi
RUN mkdir /tmp/obj
RUN chmod 777 /tmp/obj
