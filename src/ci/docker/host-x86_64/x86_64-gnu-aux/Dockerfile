FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
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
  gnupg \
  apt-utils \
  wget \
  fonts-ipafont-gothic \
  fonts-wqy-zenhei \
  fonts-thai-tlwg \
  fonts-kacst \
  fonts-freefont-ttf \

RUN curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
RUN apt install -y nodejs

# This part is to install the requirements to run rustdoc GUI tests
#
# To do so we need:
#  * a chrome instance (we can extend it to firefox)
#  * cloning the browser-UI-test repository from https://github.com/GuillaumeGomez/browser-UI-test/
#  * cloning the test-rust-docs-ui repository from https://github.com/GuillaumeGomez/test-rust-docs-ui
#
# The browser-UI-test repository is a framework to make it as easy as possible to write GUI tests
# where the test-rust-docs-ui repository contains the tests specific to rustdoc GUI.
#
# NOte that the two repositories clone part is handled by bootstrap.
#
# To prevent unwanted changes breaking tests for everyone, both repositories will target a specfic
# commit that will need to be updated when the rustdoc GUI will be updated.
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list'

RUN apt-get update && apt-get install -y --no-install-recommends \
  google-chrome-unstable \
  && rm -rf /var/lib/apt/lists/*

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_CONFIGURE_ARGS --build=x86_64-unknown-linux-gnu
ENV RUST_CHECK_TARGET check-aux-and-gui
