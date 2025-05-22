FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
COPY scripts/android-base-apt-get.sh /scripts/
RUN sh /scripts/android-base-apt-get.sh

COPY scripts/android-ndk.sh /scripts/
RUN . /scripts/android-ndk.sh && \
    download_ndk android-ndk-r26d-linux.zip

RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
  libgl1 \
  libglx-mesa0 \
  libpulse0 \
  libstdc++6:i386 \
  openjdk-8-jre-headless \
  tzdata \
  wget \
  python3

COPY scripts/android-sdk.sh /scripts/
COPY scripts/android-sdk-manager.py /scripts/
COPY host-x86_64/arm-android/android-sdk.lock /android/sdk/android-sdk.lock
RUN /scripts/android-sdk.sh

ENV PATH=$PATH:/android/sdk/emulator
ENV PATH=$PATH:/android/sdk/tools
ENV PATH=$PATH:/android/sdk/platform-tools
ENV PATH=$PATH:/android/ndk/toolchains/llvm/prebuilt/linux-x86_64/bin

ENV TARGETS=arm-linux-androideabi

ENV RUST_CONFIGURE_ARGS --android-ndk=/android/ndk/

ENV SCRIPT python3 ../x.py --stage 2 test --host='' --target $TARGETS

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY scripts/android-start-emulator.sh /scripts/
ENTRYPOINT ["/scripts/android-start-emulator.sh"]
