FROM rust-ci-base

# ndk
COPY scripts/android-ndk.sh /scripts/
RUN . /scripts/android-ndk.sh && \
    download_and_make_toolchain android-ndk-r13b-linux-x86_64.zip x86_64 21

# env
ENV PATH=$PATH:/android/ndk/x86_64-21/bin

ENV DEP_Z_ROOT=/android/ndk/x86_64-21/sysroot/usr/

ENV HOSTS=x86_64-linux-android

ENV RUST_CONFIGURE_ARGS \
      --host=$HOSTS \
      --target=$HOSTS \
      --x86_64-linux-android-ndk=/android/ndk/x86_64-21 \
      --disable-rpath \
      --enable-extended \
      --enable-cargo-openssl-static

ENV SCRIPT python2.7 ../x.py dist --target $HOSTS --host $HOSTS

# init
ENTRYPOINT ["/usr/bin/dumb-init", "--"]
