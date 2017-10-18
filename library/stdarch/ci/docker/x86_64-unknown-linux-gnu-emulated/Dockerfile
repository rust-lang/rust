FROM ubuntu:17.04
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  libc6-dev \
  file \
  make \
  ca-certificates \
  curl \
  bzip2

RUN curl https://s3-us-west-1.amazonaws.com/rust-lang-ci2/rust-ci-mirror/sde-external-8.9.0-2017-08-06-lin.tar.bz2 | \
  tar xjf -
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="/sde-external-8.9.0-2017-08-06-lin/sde64 --"
