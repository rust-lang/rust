FROM ubuntu:16.04

COPY scripts/android-base-apt-get.sh /scripts/
RUN sh /scripts/android-base-apt-get.sh

# ndk
COPY scripts/android-ndk.sh /scripts/
RUN . /scripts/android-ndk.sh && \
    download_ndk android-ndk-r15c-linux-x86_64.zip && \
    make_standalone_toolchain arm 14 && \
    make_standalone_toolchain x86 14 && \
    make_standalone_toolchain arm64 21 && \
    make_standalone_toolchain x86_64 21 && \
    remove_ndk

# env
ENV TARGETS=arm-linux-androideabi
ENV TARGETS=$TARGETS,armv7-linux-androideabi
ENV TARGETS=$TARGETS,thumbv7neon-linux-androideabi
ENV TARGETS=$TARGETS,i686-linux-android
ENV TARGETS=$TARGETS,aarch64-linux-android
ENV TARGETS=$TARGETS,x86_64-linux-android

ENV RUST_CONFIGURE_ARGS \
      --enable-extended \
      --arm-linux-androideabi-ndk=/android/ndk/arm-14 \
      --armv7-linux-androideabi-ndk=/android/ndk/arm-14 \
      --thumbv7neon-linux-androideabi-ndk=/android/ndk/arm-14 \
      --i686-linux-android-ndk=/android/ndk/x86-14 \
      --aarch64-linux-android-ndk=/android/ndk/arm64-21 \
      --x86_64-linux-android-ndk=/android/ndk/x86_64-21 \
      --disable-docs

ENV SCRIPT python2.7 ../x.py dist --target $TARGETS

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh
