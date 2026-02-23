FROM ghcr.io/rust-lang/ubuntu:22.04

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

WORKDIR /build

COPY scripts/musl-toolchain.sh /build/
# We need to mitigate rust-lang/rust#34978 when compiling musl itself as well
RUN CFLAGS="-Wa,--compress-debug-sections=none -Wl,--compress-debug-sections=none" \
    CXXFLAGS="-Wa,--compress-debug-sections=none -Wl,--compress-debug-sections=none" \
    bash musl-toolchain.sh aarch64 && rm -rf build

WORKDIR /tmp

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
