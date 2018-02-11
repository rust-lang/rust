FROM ubuntu:16.04

COPY scripts/android-base-apt-get.sh /scripts/
RUN sh /scripts/android-base-apt-get.sh

COPY scripts/android-ndk.sh /scripts/
RUN . /scripts/android-ndk.sh && \
    download_ndk android-ndk-r15c-linux-x86_64.zip && \
    make_standalone_toolchain arm 14 && \
    make_standalone_toolchain arm 21 && \
    remove_ndk

RUN chmod 777 /android/ndk && \
    ln -s /android/ndk/arm-21 /android/ndk/arm

ENV PATH=$PATH:/android/ndk/arm-14/bin

ENV DEP_Z_ROOT=/android/ndk/arm-14/sysroot/usr/

ENV HOSTS=armv7-linux-androideabi

ENV RUST_CONFIGURE_ARGS \
      --armv7-linux-androideabi-ndk=/android/ndk/arm \
      --disable-rpath \
      --enable-extended \
      --enable-cargo-openssl-static

# We support api level 14, but api level 21 is required to build llvm. To
# overcome this problem we use a ndk with api level 21 to build llvm and then
# switch to a ndk with api level 14 to complete the build. When the linker is
# invoked there are missing symbols (like sigsetempty, not available with api
# level 14), the default linker behavior is to generate an error, to allow the
# build to finish we use --warn-unresolved-symbols. Note that the missing
# symbols does not affect std, only the compiler (llvm) and cargo (openssl).
ENV SCRIPT \
  python2.7 ../x.py build src/llvm --host $HOSTS --target $HOSTS && \
  (export RUSTFLAGS="\"-C link-arg=-Wl,--warn-unresolved-symbols\""; \
    rm /android/ndk/arm && \
    ln -s /android/ndk/arm-14 /android/ndk/arm && \
    python2.7 ../x.py dist --host $HOSTS --target $HOSTS)

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh
