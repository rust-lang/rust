FROM ubuntu:26.04

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  clang-22 \
  llvm-22 \
  lld-22

COPY scripts/crosstool-ng-git.sh /scripts/
COPY scripts/crosstool-ng-sha256-20260705.diff /scripts/
RUN sh /scripts/crosstool-ng-git.sh

COPY scripts/rustbuild-setup.sh /scripts/
RUN sh /scripts/rustbuild-setup.sh
WORKDIR /tmp

COPY scripts/crosstool-ng-build.sh /scripts/
COPY host-x86_64/dist-loongarch64-musl/loongarch64-unknown-linux-musl.defconfig /tmp/crosstool.defconfig
RUN /scripts/crosstool-ng-build.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV PATH=/usr/lib/llvm-22/bin:$PATH:/x-tools/loongarch64-unknown-linux-musl/bin

# --no-rosegment keeps read-only code in the first LOAD segment, matching the
# segment layout produced by GNU ld. This allows Linux to make better use of
# file-backed PMD mappings and reduces iTLB misses.
ENV CLANG_FLAGS="--target=loongarch64-unknown-linux-musl -fuse-ld=lld \
    --gcc-toolchain=/x-tools/loongarch64-unknown-linux-musl \
    --sysroot=/x-tools/loongarch64-unknown-linux-musl/loongarch64-unknown-linux-musl/sysroot \
    -Wl,--no-rosegment"

ENV CC=clang \
    CXX=clang++ \
    CC_loongarch64_unknown_linux_musl=clang \
    CXX_loongarch64_unknown_linux_musl=clang++ \
    CFLAGS_loongarch64_unknown_linux_musl="$CLANG_FLAGS -mcmodel=medium" \
    CXXFLAGS_loongarch64_unknown_linux_musl="$CLANG_FLAGS -mcmodel=medium" \
    AR_loongarch64_unknown_linux_musl=llvm-ar \
    RANLIB_loongarch64_unknown_linux_musl=llvm-ranlib \
    CARGO_TARGET_LOONGARCH64_UNKNOWN_LINUX_MUSL_LINKER=clang \
    CARGO_TARGET_LOONGARCH64_UNKNOWN_LINUX_MUSL_RUSTFLAGS="-Clink-arg=${CLANG_FLAGS// / -Clink-arg=}"

ENV HOSTS=loongarch64-unknown-linux-musl

ENV RUST_CONFIGURE_ARGS="--enable-extended \
      --enable-full-tools \
      --enable-profiler \
      --enable-sanitizers \
      --disable-docs \
      --set build.allocator=jemalloc \
      --set llvm.link-shared=true \
      --set llvm.thin-lto=true \
      --set rust.lto=thin \
      --set rust.codegen-units=1 \
      --set target.loongarch64-unknown-linux-musl.crt-static=false \
      --musl-root-loongarch64=/x-tools/loongarch64-unknown-linux-musl/loongarch64-unknown-linux-musl/sysroot/usr"

ENV SCRIPT="python3 ../x.py dist --host $HOSTS --target $HOSTS"
