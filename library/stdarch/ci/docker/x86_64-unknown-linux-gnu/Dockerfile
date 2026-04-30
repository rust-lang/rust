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

RUN wget http://ci-mirrors.rust-lang.org/sde-external-10.8.0-2026-03-15-lin.tar.xz -O sde.tar.xz
RUN mkdir intel-sde
RUN tar -xJf sde.tar.xz --strip-components=1 -C intel-sde

RUN wget https://mirrors.edge.kernel.org/pub/tools/llvm/files/llvm-22.1.4-x86_64.tar.gz -O llvm.tar.xz
RUN mkdir llvm
RUN tar -xvf llvm.tar.xz --strip-components=1 -C llvm

ENV PATH="/llvm/bin:$PATH"

ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="/intel-sde/sde64 \
            -cpuid-in /checkout/ci/docker/x86_64-unknown-linux-gnu/cpuid.def \
            -rtm-mode full -tsx --"
# SDE doesn't support AMD extensions
# FIXME: find a way to test these
ENV STDARCH_TEST_SKIP_FEATURE="sse4a,tbm,xop"
