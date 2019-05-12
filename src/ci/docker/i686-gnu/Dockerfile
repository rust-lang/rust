FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++-multilib \
  make \
  file \
  curl \
  ca-certificates \
  python2.7 \
  git \
  cmake \
  sudo \
  gdb \
  xz-utils


COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_CONFIGURE_ARGS --build=i686-unknown-linux-gnu
# Exclude some tests that are unlikely to be platform specific, to speed up
# this slow job.
ENV SCRIPT python2.7 ../x.py test \
  --exclude src/bootstrap \
  --exclude src/test/rustdoc-js \
  --exclude src/tools/error_index_generator \
  --exclude src/tools/linkchecker
