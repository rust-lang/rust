FROM ubuntu:16.04

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
  libssl-dev \
  sudo \
  xz-utils \
  pkg-config \
  libgl1-mesa-dev \
  llvm-dev \
  libfreetype6-dev \
  libexpat1-dev \
  libexpat1-dev \
  gnupg \
  apt-utils \
  wget \
  fonts-ipafont-gothic \
  fonts-wqy-zenhei \
  fonts-thai-tlwg \
  fonts-kacst \
  fonts-freefont-ttf \
  libxss1 \
  libxtst6

RUN curl -sL https://nodejs.org/dist/v14.4.0/node-v14.4.0-linux-x64.tar.xz | tar -xJ
ENV PATH="/node-v14.4.0-linux-x64/bin:${PATH}"

# Install required dependencies from browser-UI-test framework
# For now, we need to use `--unsafe-perm=true` to go around an issue when npm tries
# to create a new folder. For reference:
# https://github.com/puppeteer/puppeteer/issues/375
RUN npm install browser-ui-test -g --unsafe-perm=true

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY scripts/cmake.sh /scripts/
RUN /scripts/cmake.sh

ENV RUST_CONFIGURE_ARGS --build=x86_64-unknown-linux-gnu
ENV RUST_CHECK_TARGET check-aux-and-gui
