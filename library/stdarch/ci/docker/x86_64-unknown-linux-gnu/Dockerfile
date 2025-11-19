FROM ubuntu:25.10
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  libc6-dev \
  file \
  make \
  ca-certificates \
  wget \
  xz-utils \
  clang \
  libstdc++-14-dev \
  build-essential \
  lld

RUN wget http://ci-mirrors.rust-lang.org/stdarch/sde-external-9.58.0-2025-06-16-lin.tar.xz -O sde.tar.xz
RUN mkdir intel-sde
RUN tar -xJf sde.tar.xz --strip-components=1 -C intel-sde
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="/intel-sde/sde64 \
            -cpuid-in /checkout/ci/docker/x86_64-unknown-linux-gnu/cpuid.def \
            -rtm-mode full -tsx --"
# SDE doesn't support AMD extensions
# FIXME: find a way to test these
ENV STDARCH_TEST_SKIP_FEATURE="sse4a,tbm,xop"
