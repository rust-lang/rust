FROM ubuntu:16.04

RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
  file \
  make \
  curl \
  ca-certificates \
  python \
  unzip \
  expect \
  openjdk-9-jre \
  libstdc++6:i386 \
  libpulse0 \
  gcc \
  libc6-dev

WORKDIR /android/
COPY android* /android/

ENV ANDROID_ARCH=arm
ENV PATH=$PATH:/android/ndk-$ANDROID_ARCH/bin:/android/sdk/tools:/android/sdk/platform-tools

RUN sh /android/android-install-ndk.sh $ANDROID_ARCH
RUN sh /android/android-install-sdk.sh $ANDROID_ARCH
RUN mv /root/.android /tmp
RUN chmod 777 -R /tmp/.android
RUN chmod 755 /android/sdk/tools/* /android/sdk/emulator/qemu/linux-x86_64/*

ENV PATH=$PATH:/rust/bin \
    CARGO_TARGET_ARM_LINUX_ANDROIDEABI_LINKER=arm-linux-androideabi-gcc \
    CARGO_TARGET_ARM_LINUX_ANDROIDEABI_RUNNER=/tmp/runtest \
    OBJDUMP=arm-linux-androideabi-objdump \
    HOME=/tmp

ADD runtest-android.rs /tmp/runtest.rs
ENTRYPOINT [ \
  "bash", \
  "-c", \
  # set SHELL so android can detect a 64bits system, see
  # http://stackoverflow.com/a/41789144
  "SHELL=/bin/dash /android/sdk/emulator/emulator @arm -no-window & \
   rustc /tmp/runtest.rs -o /tmp/runtest && \
   exec \"$@\"", \
  "--" \
]
