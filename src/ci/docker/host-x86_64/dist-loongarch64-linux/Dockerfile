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
COPY host-x86_64/dist-loongarch64-linux/loongarch64-unknown-linux-gnu.defconfig /tmp/crosstool.defconfig
RUN /scripts/crosstool-ng-build.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV PATH=/usr/lib/llvm-22/bin:$PATH:/x-tools/loongarch64-unknown-linux-gnu/bin

# --no-rosegment keeps read-only code in the first LOAD segment, matching the
# segment layout produced by GNU ld. This allows Linux to make better use of
# file-backed PMD mappings and reduces iTLB misses.
ENV CLANG_FLAGS="--target=loongarch64-unknown-linux-gnu -fuse-ld=lld \
    --gcc-toolchain=/x-tools/loongarch64-unknown-linux-gnu \
    --sysroot=/x-tools/loongarch64-unknown-linux-gnu/loongarch64-unknown-linux-gnu/sysroot \
    -Wl,--no-rosegment"

ENV CC=clang \
    CXX=clang++ \
    CC_loongarch64_unknown_linux_gnu=clang \
    CXX_loongarch64_unknown_linux_gnu=clang++ \
    CFLAGS_loongarch64_unknown_linux_gnu="$CLANG_FLAGS -mcmodel=medium" \
    CXXFLAGS_loongarch64_unknown_linux_gnu="$CLANG_FLAGS -mcmodel=medium" \
    AR_loongarch64_unknown_linux_gnu=llvm-ar \
    RANLIB_loongarch64_unknown_linux_gnu=llvm-ranlib \
    CARGO_TARGET_LOONGARCH64_UNKNOWN_LINUX_GNU_LINKER=clang \
    CARGO_TARGET_LOONGARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-Clink-arg=${CLANG_FLAGS// / -Clink-arg=}"

# We re-use the Linux toolchain for bare-metal, because upstream bare-metal
# target support for LoongArch is only available from GCC 14+.
#
# See: https://github.com/gcc-mirror/gcc/commit/976f4f9e4770
ENV CC_loongarch64_unknown_none=loongarch64-unknown-linux-gnu-gcc \
    AR_loongarch64_unknown_none=loongarch64-unknown-linux-gnu-ar \
    CXX_loongarch64_unknown_none=loongarch64-unknown-linux-gnu-g++ \
    CFLAGS_loongarch64_unknown_none="-ffreestanding -mabi=lp64d" \
    CXXFLAGS_loongarch64_unknown_none="-ffreestanding -mabi=lp64d" \
    CC_loongarch64_unknown_none_softfloat=loongarch64-unknown-linux-gnu-gcc \
    AR_loongarch64_unknown_none_softfloat=loongarch64-unknown-linux-gnu-ar \
    CXX_loongarch64_unknown_none_softfloat=loongarch64-unknown-linux-gnu-g++ \
    CFLAGS_loongarch64_unknown_none_softfloat="-ffreestanding -mabi=lp64s -mfpu=none" \
    CXXFLAGS_loongarch64_unknown_none_softfloat="-ffreestanding -mabi=lp64s -mfpu=none" \
    CC_loongarch32_unknown_none=loongarch64-unknown-linux-gnu-gcc \
    AR_loongarch32_unknown_none=loongarch64-unknown-linux-gnu-ar \
    CXX_loongarch32_unknown_none=loongarch64-unknown-linux-gnu-g++ \
    CFLAGS_loongarch32_unknown_none="-ffreestanding -march=la32rv1.0 -mabi=ilp32d" \
    CXXFLAGS_loongarch32_unknown_none="-ffreestanding -march=la32rv1.0 -mabi=ilp32d" \
    CC_loongarch32_unknown_none_softfloat=loongarch64-unknown-linux-gnu-gcc \
    AR_loongarch32_unknown_none_softfloat=loongarch64-unknown-linux-gnu-ar \
    CXX_loongarch32_unknown_none_softfloat=loongarch64-unknown-linux-gnu-g++ \
    CFLAGS_loongarch32_unknown_none_softfloat="-ffreestanding -march=la32rv1.0 -mabi=ilp32s -mfpu=none" \
    CXXFLAGS_loongarch32_unknown_none_softfloat="-ffreestanding -march=la32rv1.0 -mabi=ilp32s -mfpu=none"

ENV HOSTS=loongarch64-unknown-linux-gnu
ENV TARGETS=$HOSTS
ENV TARGETS=$TARGETS,loongarch64-unknown-none
ENV TARGETS=$TARGETS,loongarch64-unknown-none-softfloat
ENV TARGETS=$TARGETS,loongarch32-unknown-none
ENV TARGETS=$TARGETS,loongarch32-unknown-none-softfloat

ENV RUST_CONFIGURE_ARGS="--enable-extended \
      --enable-full-tools \
      --enable-profiler \
      --enable-sanitizers \
      --disable-docs \
      --set build.allocator=jemalloc \
      --set llvm.link-shared=true \
      --set llvm.thin-lto=true \
      --set rust.lto=thin \
      --set rust.codegen-units=1"

ENV SCRIPT="python3 ../x.py dist --host $HOSTS --target $TARGETS"
