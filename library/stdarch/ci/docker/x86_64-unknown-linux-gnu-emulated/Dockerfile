FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  libc6-dev \
  file \
  make \
  ca-certificates \
  wget \
  xz-utils

RUN wget https://downloadmirror.intel.com/843185/sde-external-9.48.0-2024-11-25-lin.tar.xz
RUN tar -xJf sde-external-9.48.0-2024-11-25-lin.tar.xz
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="/sde-external-9.48.0-2024-11-25-lin/sde64 \
            -cpuid-in /checkout/ci/docker/x86_64-unknown-linux-gnu-emulated/cpuid.def \
            -rtm-mode full -tsx --"
