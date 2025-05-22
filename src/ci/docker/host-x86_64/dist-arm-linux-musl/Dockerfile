FROM ghcr.io/rust-lang/ubuntu:22.04

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

COPY scripts/crosstool-ng.sh /scripts/
RUN sh /scripts/crosstool-ng.sh

WORKDIR /build

COPY scripts/musl-toolchain.sh /build/
# We need to mitigate rust-lang/rust#34978 when compiling musl itself as well
RUN CFLAGS="-Wa,--compress-debug-sections=none -Wl,--compress-debug-sections=none" \
    CXXFLAGS="-Wa,--compress-debug-sections=none -Wl,--compress-debug-sections=none" \
    bash musl-toolchain.sh aarch64 && rm -rf build

COPY scripts/rustbuild-setup.sh /scripts/
RUN sh /scripts/rustbuild-setup.sh
WORKDIR /tmp

COPY scripts/crosstool-ng-build.sh /scripts/
COPY host-x86_64/dist-arm-linux-musl/arm-linux-musl.defconfig /tmp/crosstool.defconfig
RUN /scripts/crosstool-ng-build.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV HOSTS=aarch64-unknown-linux-musl

ENV RUST_CONFIGURE_ARGS \
      --enable-full-tools \
      --disable-docs \
      --musl-root-aarch64=/usr/local/aarch64-linux-musl \
      --enable-sanitizers \
      --enable-profiler \
      --set target.aarch64-unknown-linux-musl.crt-static=false
ENV SCRIPT python3 ../x.py dist --host $HOSTS --target $HOSTS
