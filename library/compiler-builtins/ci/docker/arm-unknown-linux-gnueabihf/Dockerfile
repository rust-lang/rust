ARG IMAGE=ubuntu:25.04
FROM $IMAGE

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc libc6-dev ca-certificates \
    gcc-arm-linux-gnueabihf libc6-dev-armhf-cross qemu-user-static

ENV TOOLCHAIN_PREFIX=arm-linux-gnueabihf-
ENV CARGO_TARGET_ARM_UNKNOWN_LINUX_GNUEABIHF_LINKER="$TOOLCHAIN_PREFIX"gcc \
    CARGO_TARGET_ARM_UNKNOWN_LINUX_GNUEABIHF_RUNNER=qemu-arm-static \
    AR_arm_unknown_linux_gnueabihf="$TOOLCHAIN_PREFIX"ar \
    CC_arm_unknown_linux_gnueabihf="$TOOLCHAIN_PREFIX"gcc \
    QEMU_LD_PREFIX=/usr/arm-linux-gnueabihf \
    RUST_TEST_THREADS=1
