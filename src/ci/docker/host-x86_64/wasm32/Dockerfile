FROM ubuntu:20.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
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
  bzip2 \
  && rm -rf /var/lib/apt/lists/*

COPY scripts/emscripten.sh /scripts/
RUN bash /scripts/emscripten.sh

COPY scripts/cmake.sh /scripts/
RUN /scripts/cmake.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

# emcc seems to need python to specifically be "python" and not "python3"
RUN ln `which python3` /usr/bin/python

ENV PATH=$PATH:/emsdk-portable
ENV PATH=$PATH:/emsdk-portable/upstream/emscripten/

# Rust's build system requires NodeJS to be in the path, but the directory in
# which emsdk stores it contains the version number. This caused breakages in
# the past when emsdk bumped the node version causing the path to point to a
# missing directory.
#
# To avoid the problem this symlinks the latest NodeJs version available to
# "latest", and adds that to the path.
RUN ln -s /emsdk-portable/node/$(ls /emsdk-portable/node | sort -V | tail -n 1) \
          /emsdk-portable/node/latest
ENV PATH=$PATH:/emsdk-portable/node/latest/bin/

ENV BINARYEN_ROOT=/emsdk-portable/upstream/
ENV EMSDK=/emsdk-portable
ENV EM_CONFIG=/emsdk-portable/.emscripten
ENV EM_CACHE=/emsdk-portable/upstream/emscripten/cache

ENV TARGETS=wasm32-unknown-emscripten

# Use -O1 optimizations in the link step to reduce time spent optimizing.
ENV EMCC_CFLAGS=-O1

COPY static/gitconfig /etc/gitconfig

# Emscripten installation is user-specific
ENV NO_CHANGE_USER=1

# Exclude library/alloc due to OOM in benches.
ENV SCRIPT python3 ../x.py test --stage 2 --host='' --target $TARGETS \
    --exclude library/alloc
