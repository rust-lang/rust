FROM ubuntu:16.04

COPY scripts/android-base-apt-get.sh /scripts/
RUN sh /scripts/android-base-apt-get.sh

COPY scripts/android-ndk.sh /scripts/
RUN . /scripts/android-ndk.sh && \
    download_and_make_toolchain android-ndk-r15c-linux-x86_64.zip arm 14

# Note:
# Do not upgrade to `openjdk-9-jre-headless`, as it will cause certificate error
# when installing the Android SDK (see PR #45193). This is unfortunate, but
# every search result suggested either disabling HTTPS or replacing JDK 9 by
# JDK 8 as the solution (e.g. https://stackoverflow.com/q/41421340). :|
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
  libgl1-mesa-glx \
  libpulse0 \
  libstdc++6:i386 \
  openjdk-8-jre-headless \
  tzdata

COPY scripts/android-sdk.sh /scripts/
RUN . /scripts/android-sdk.sh && \
    download_and_create_avd 4333796 armeabi-v7a 18 5264690

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
