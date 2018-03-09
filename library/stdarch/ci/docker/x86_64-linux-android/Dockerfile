FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  gcc \
  libc-dev \
  python \
  unzip \
  file \
  make

WORKDIR /android/
ENV ANDROID_ARCH=x86_64
COPY android-install-ndk.sh /android/
RUN sh /android/android-install-ndk.sh $ANDROID_ARCH

# We do not run x86_64-linux-android tests on an android emulator.
# See ci/android-sysimage.sh for informations about how tests are run.
COPY android-sysimage.sh /android/
RUN bash /android/android-sysimage.sh x86_64 x86_64-24_r07.zip

ENV PATH=$PATH:/rust/bin:/android/ndk-$ANDROID_ARCH/bin \
    CARGO_TARGET_X86_64_LINUX_ANDROID_LINKER=x86_64-linux-android-gcc \
    CC_x86_64_linux_android=x86_64-linux-android-gcc \
    CXX_x86_64_linux_android=x86_64-linux-android-g++ \
    OBJDUMP=x86_64-linux-android-objdump \
    HOME=/tmp
