FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python3 \
  git \
  cmake \
  sudo \
  gdb \
  xz-utils \
  libssl-dev \
  pkg-config \
  mingw-w64

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY host-x86_64/mingw-check/validate-toolstate.sh /scripts/

ENV RUN_CHECK_WITH_PARALLEL_QUERIES 1
ENV SCRIPT python3 ../x.py --stage 2 test src/tools/expand-yaml-anchors && \
           python3 ../x.py check --target=i686-pc-windows-gnu --host=i686-pc-windows-gnu && \
           python3 ../x.py build --stage 0 src/tools/build-manifest && \
           python3 ../x.py test --stage 0 src/tools/compiletest && \
           python3 ../x.py test --stage 2 src/tools/tidy && \
           python3 ../x.py doc --stage 0 library/std && \
           /scripts/validate-toolstate.sh
