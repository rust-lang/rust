FROM mato/rumprun-toolchain-hw-x86_64
USER root
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
  qemu
ENV PATH=$PATH:/rust/bin
