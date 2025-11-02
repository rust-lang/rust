ARG IMAGE=ubuntu:25.04
FROM $IMAGE

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc libc6-dev ca-certificates \
    gcc-arm-linux-gnueabi libc6-dev-armel-cross qemu-user-static

ENV TOOLCHAIN_PREFIX=arm-linux-gnueabi-
ENV CARGO_TARGET_ARM_UNKNOWN_LINUX_GNUEABI_LINKER="$TOOLCHAIN_PREFIX"gcc \
    CARGO_TARGET_ARM_UNKNOWN_LINUX_GNUEABI_RUNNER=qemu-arm-static \
    AR_arm_unknown_linux_gnueabi="$TOOLCHAIN_PREFIX"ar \
    CC_arm_unknown_linux_gnueabi="$TOOLCHAIN_PREFIX"gcc \
    QEMU_LD_PREFIX=/usr/arm-linux-gnueabi \
    RUST_TEST_THREADS=1
