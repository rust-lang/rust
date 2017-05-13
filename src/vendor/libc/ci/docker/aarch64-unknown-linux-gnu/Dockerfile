FROM ubuntu:16.10
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
  gcc libc6-dev ca-certificates \
  gcc-aarch64-linux-gnu libc6-dev-arm64-cross qemu-user
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
    PATH=$PATH:/rust/bin
