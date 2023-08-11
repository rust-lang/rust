FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python2.7 \
  python3 \
  python3-pip \
  python3-pkg-resources \
  git \
  cmake \
  sudo \
  gdb \
  xz-utils \
  libssl-dev \
  pkg-config \
  mingw-w64 \
  && rm -rf /var/lib/apt/lists/*

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY host-x86_64/mingw-check/reuse-requirements.txt /tmp/
RUN pip3 install --no-deps --no-cache-dir --require-hashes -r /tmp/reuse-requirements.txt \
    && pip3 install virtualenv

COPY host-x86_64/mingw-check/validate-toolstate.sh /scripts/
COPY host-x86_64/mingw-check/validate-error-codes.sh /scripts/

# NOTE: intentionally uses python2 for x.py so we can test it still works.
# validate-toolstate only runs in our CI, so it's ok for it to only support python3.
ENV SCRIPT TIDY_PRINT_DIFF=1 python2.7 ../x.py test \
           --stage 0 src/tools/tidy tidyselftest --extra-checks=py:lint
