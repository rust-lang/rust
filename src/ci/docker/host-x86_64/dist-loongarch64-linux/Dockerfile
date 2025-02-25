FROM ubuntu:22.04

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

COPY scripts/crosstool-ng.sh /scripts/
RUN sh /scripts/crosstool-ng.sh

COPY scripts/rustbuild-setup.sh /scripts/
RUN sh /scripts/rustbuild-setup.sh
WORKDIR /tmp

COPY scripts/crosstool-ng-build.sh /scripts/
COPY host-x86_64/dist-loongarch64-linux/loongarch64-unknown-linux-gnu.defconfig /tmp/crosstool.defconfig
RUN /scripts/crosstool-ng-build.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV PATH=$PATH:/x-tools/loongarch64-unknown-linux-gnu/bin

ENV CC_loongarch64_unknown_linux_gnu=loongarch64-unknown-linux-gnu-gcc \
    AR_loongarch64_unknown_linux_gnu=loongarch64-unknown-linux-gnu-ar \
    CXX_loongarch64_unknown_linux_gnu=loongarch64-unknown-linux-gnu-g++

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
    CXXFLAGS_loongarch64_unknown_none_softfloat="-ffreestanding -mabi=lp64s -mfpu=none"

ENV HOSTS=loongarch64-unknown-linux-gnu
ENV TARGETS=$HOSTS
ENV TARGETS=$TARGETS,loongarch64-unknown-none
ENV TARGETS=$TARGETS,loongarch64-unknown-none-softfloat

ENV RUST_CONFIGURE_ARGS \
      --enable-extended \
      --enable-full-tools \
      --enable-profiler \
      --enable-sanitizers \
      --disable-docs

ENV SCRIPT python3 ../x.py dist --host $HOSTS --target $TARGETS
