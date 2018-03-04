FROM ubuntu:16.04

COPY scripts/android-base-apt-get.sh /scripts/
RUN sh /scripts/android-base-apt-get.sh

COPY scripts/android-ndk.sh /scripts/
RUN . /scripts/android-ndk.sh && \
    download_ndk android-ndk-r15c-linux-x86_64.zip && \
    make_standalone_toolchain x86 14 && \
    make_standalone_toolchain x86 21 && \
    remove_ndk

RUN chmod 777 /android/ndk && \
    ln -s /android/ndk/x86-21 /android/ndk/x86

ENV PATH=$PATH:/android/ndk/x86-14/bin

ENV DEP_Z_ROOT=/android/ndk/x86-14/sysroot/usr/

ENV HOSTS=i686-linux-android

ENV RUST_CONFIGURE_ARGS \
      --i686-linux-android-ndk=/android/ndk/x86 \
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
    rm /android/ndk/x86 && \
    ln -s /android/ndk/x86-14 /android/ndk/x86 && \
    python2.7 ../x.py dist --host $HOSTS --target $HOSTS)

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh
