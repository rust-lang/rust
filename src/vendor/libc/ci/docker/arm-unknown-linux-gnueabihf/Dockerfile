FROM ubuntu:16.10
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
  gcc libc6-dev ca-certificates \
  gcc-arm-linux-gnueabihf libc6-dev-armhf-cross qemu-user
ENV CARGO_TARGET_ARM_UNKNOWN_LINUX_GNUEABIHF_LINKER=arm-linux-gnueabihf-gcc \
    PATH=$PATH:/rust/bin
