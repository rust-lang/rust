FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
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
  tidy

# Install dependencies for chromium browser
RUN apt-get install -y \
  gconf-service \
  libasound2 \
  libatk1.0-0 \
  libatk-bridge2.0-0 \
  libc6 \
  libcairo2 \
  libcups2 \
  libdbus-1-3 \
  libexpat1 \
  libfontconfig1 \
  libgbm1 \
  libgcc1 \
  libgconf-2-4 \
  libgdk-pixbuf2.0-0 \
  libglib2.0-0 \
  libgtk-3-0 \
  libnspr4 \
  libpango-1.0-0 \
  libpangocairo-1.0-0 \
  libstdc++6 \
  libx11-6 \
  libx11-xcb1 \
  libxcb1 \
  libxcomposite1 \
  libxcursor1 \
  libxdamage1 \
  libxext6 \
  libxfixes3 \
  libxi6 \
  libxrandr2 \
  libxrender1 \
  libxss1 \
  libxtst6 \
  fonts-liberation \
  libappindicator1 \
  libnss3 \
  lsb-release \
  xdg-utils \
  wget

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY host-x86_64/x86_64-gnu-tools/checktools.sh /tmp/

RUN curl -sL https://nodejs.org/dist/v14.20.0/node-v14.20.0-linux-x64.tar.xz | tar -xJ
ENV NODE_FOLDER=/node-v14.20.0-linux-x64/bin
ENV PATH="$NODE_FOLDER:${PATH}"

COPY host-x86_64/x86_64-gnu-tools/browser-ui-test.version /tmp/

# For now, we need to use `--unsafe-perm=true` to go around an issue when npm tries
# to create a new folder. For reference:
# https://github.com/puppeteer/puppeteer/issues/375
#
# We also specify the version in case we need to update it to go around cache limitations.
#
# The `browser-ui-test.version` file is also used by bootstrap to emit warnings in case
# the local version of the package is different than the one used by the CI.
RUN npm install -g browser-ui-test@$(head -n 1 /tmp/browser-ui-test.version) --unsafe-perm=true

ENV RUST_CONFIGURE_ARGS \
  --build=x86_64-unknown-linux-gnu \
  --save-toolstates=/tmp/toolstate/toolstates.json

ENV SCRIPT /tmp/checktools.sh ../x.py && \
  NODE_PATH=`npm root -g` python3 ../x.py test src/test/rustdoc-gui --stage 2 \
    --test-args "'--no-sandbox --jobs 1'"
