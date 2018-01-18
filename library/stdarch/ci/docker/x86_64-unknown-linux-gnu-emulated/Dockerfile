FROM ubuntu:17.10
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  libc6-dev \
  file \
  make \
  ca-certificates \
  wget \
  bzip2

RUN wget https://github.com/gnzlbg/intel_sde/raw/master/sde-external-8.12.0-2017-10-23-lin.tar.bz2
RUN tar -xjf sde-external-8.12.0-2017-10-23-lin.tar.bz2
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="/sde-external-8.12.0-2017-10-23-lin/sde64 --"
