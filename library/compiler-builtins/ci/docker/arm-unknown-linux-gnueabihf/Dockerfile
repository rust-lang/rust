FROM ubuntu:16.04
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
  gcc libc6-dev ca-certificates \
  gcc-arm-linux-gnueabihf libc6-dev-armhf-cross qemu-user-static
ENV CARGO_TARGET_ARM_UNKNOWN_LINUX_GNUEABIHF_LINKER=arm-linux-gnueabihf-gcc \
    QEMU_LD_PREFIX=/usr/arm-linux-gnueabihf \
    RUST_TEST_THREADS=1
