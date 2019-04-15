FROM ubuntu:16.04

COPY scripts/android-base-apt-get.sh /scripts/
RUN sh /scripts/android-base-apt-get.sh

COPY scripts/android-ndk.sh /scripts/
RUN . /scripts/android-ndk.sh && \
    download_and_make_toolchain android-ndk-r15c-linux-x86_64.zip arm 14

RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
  libgl1-mesa-glx \
  libpulse0 \
  libstdc++6:i386 \
  openjdk-9-jre-headless \
  tzdata \
  wget \
  python3

COPY scripts/android-sdk.sh /scripts/
COPY scripts/android-sdk-manager.py /scripts/
COPY arm-android/android-sdk.lock /android/sdk/android-sdk.lock
RUN /scripts/android-sdk.sh

ENV PATH=$PATH:/android/sdk/emulator
ENV PATH=$PATH:/android/sdk/tools
ENV PATH=$PATH:/android/sdk/platform-tools

ENV TARGETS=arm-linux-androideabi

ENV RUST_CONFIGURE_ARGS --arm-linux-androideabi-ndk=/android/ndk/arm-14

ENV SCRIPT python2.7 ../x.py test --target $TARGETS

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY scripts/android-start-emulator.sh /scripts/
ENTRYPOINT ["/scripts/android-start-emulator.sh"]
