FROM ubuntu:18.04
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc libc6-dev ca-certificates \
    gcc-arm-linux-gnueabi libc6-dev-armel-cross qemu-user-static
ENV CARGO_TARGET_ARM_UNKNOWN_LINUX_GNUEABI_LINKER=arm-linux-gnueabi-gcc \
    CARGO_TARGET_ARM_UNKNOWN_LINUX_GNUEABI_RUNNER=qemu-arm-static \
    QEMU_LD_PREFIX=/usr/arm-linux-gnueabi \
    RUST_TEST_THREADS=1
