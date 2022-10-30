FROM ubuntu:22.04

COPY scripts/android-base-apt-get.sh /scripts/
RUN sh /scripts/android-base-apt-get.sh

# ndk
COPY scripts/android-ndk.sh /scripts/
RUN . /scripts/android-ndk.sh && \
    download_ndk android-ndk-r25b-linux.zip

# env
ENV TARGETS=arm-linux-androideabi
ENV TARGETS=$TARGETS,armv7-linux-androideabi
ENV TARGETS=$TARGETS,thumbv7neon-linux-androideabi
ENV TARGETS=$TARGETS,i686-linux-android
ENV TARGETS=$TARGETS,aarch64-linux-android
ENV TARGETS=$TARGETS,x86_64-linux-android

ENV RUST_CONFIGURE_ARGS \
      --enable-extended \
      --enable-profiler \
      --arm-linux-androideabi-ndk=/android/ndk/toolchains/llvm/prebuilt/linux-x86_64/ \
      --armv7-linux-androideabi-ndk=/android/ndk/toolchains/llvm/prebuilt/linux-x86_64/ \
      --thumbv7neon-linux-androideabi-ndk=/android/ndk/toolchains/llvm/prebuilt/linux-x86_64/ \
      --i686-linux-android-ndk=/android/ndk/toolchains/llvm/prebuilt/linux-x86_64/ \
      --aarch64-linux-android-ndk=/android/ndk/toolchains/llvm/prebuilt/linux-x86_64/ \
      --x86_64-linux-android-ndk=/android/ndk/toolchains/llvm/prebuilt/linux-x86_64/ \
      --disable-docs

ENV SCRIPT python3 ../x.py dist --host='' --target $TARGETS

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh
