FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
  g++ \
  make \
  file \
  curl \
  ca-certificates \
  python2.7 \
  git \
  cmake \
  unzip \
  sudo \
  xz-utils \
  libssl-dev \
  pkg-config

RUN curl -OL https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && \
    rm dumb-init_*.deb

RUN curl -o /usr/local/bin/sccache \
      https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/2017-05-12-sccache-x86_64-unknown-linux-musl && \
      chmod +x /usr/local/bin/sccache

ENTRYPOINT ["/usr/bin/dumb-init", "--"]

COPY android-ndk.sh /
RUN . /android-ndk.sh && \
    download_ndk android-ndk-r13b-linux-x86_64.zip && \
    make_standalone_toolchain x86 9 && \
    make_standalone_toolchain x86 21 && \
    remove_ndk

ENV PATH=$PATH:/android/ndk/x86-9/bin

ENV DEP_Z_ROOT=/android/ndk/x86-9/sysroot/usr/

ENV HOSTS=i686-linux-android

ENV RUST_CONFIGURE_ARGS \
      --host=$HOSTS \
      --target=$HOSTS \
      --i686-linux-android-ndk=/android/ndk/x86 \
      --disable-rpath \
      --enable-extended \
      --enable-cargo-openssl-static

# We support api level 9, but api level 21 is required to build llvm. To
# overcome this problem we use a ndk with api level 21 to build llvm and then
# switch to a ndk with api level 9 to complete the build. When the linker is
# invoked there are missing symbols (like sigsetempty, not available with api
# level 9), the default linker behavior is to generate an error, to allow the
# build to finish we use --warn-unresolved-symbols. Note that the missing
# symbols does not affect std, only the compiler (llvm) and cargo (openssl).
RUN chmod 777 /android/ndk && \
    ln -s /android/ndk/x86-21 /android/ndk/x86

ENV SCRIPT \
  python2.7 ../x.py build src/llvm --host $HOSTS --target $HOSTS && \
  (export RUSTFLAGS="\"-C link-arg=-Wl,--warn-unresolved-symbols\""; \
    rm /android/ndk/x86 && \
    ln -s /android/ndk/x86-9 /android/ndk/x86 && \
    python2.7 ../x.py dist --host $HOSTS --target $HOSTS)
