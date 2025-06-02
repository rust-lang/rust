FROM ghcr.io/rust-lang/ubuntu:22.04

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
  tidy \
  \
# Install dependencies for chromium browser
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
  wget \
  # libgccjit dependencies
  flex \
  libmpfr-dev \
  libgmp-dev \
  libmpc3 \
  libmpc-dev \
  && rm -rf /var/lib/apt/lists/*

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

# Fix rustc_codegen_gcc lto issues.
ENV GCC_EXEC_PREFIX="/usr/lib/gcc/"

COPY host-x86_64/x86_64-gnu-tools/checktools.sh /tmp/

COPY scripts/nodejs.sh /scripts/
RUN sh /scripts/nodejs.sh /node
ENV PATH="/node/bin:${PATH}"

COPY host-x86_64/x86_64-gnu-tools/browser-ui-test.version /tmp/

ENV RUST_CONFIGURE_ARGS \
  --build=x86_64-unknown-linux-gnu \
  --save-toolstates=/tmp/toolstate/toolstates.json \
  --enable-new-symbol-mangling

ENV HOST_TARGET x86_64-unknown-linux-gnu

# FIXME(#133381): currently rustc alt builds do *not* have rustc debug
# assertions enabled! Therefore, we cannot force download CI rustc.
#ENV FORCE_CI_RUSTC 1

COPY scripts/shared.sh /scripts/

# For now, we need to use `--unsafe-perm=true` to go around an issue when npm tries
# to create a new folder. For reference:
# https://github.com/puppeteer/puppeteer/issues/375
#
# We also specify the version in case we need to update it to go around cache limitations.
#
# The `browser-ui-test.version` file is also used by bootstrap to emit warnings in case
# the local version of the package is different than the one used by the CI.
ENV SCRIPT /tmp/checktools.sh ../x.py && \
  npm install browser-ui-test@$(head -n 1 /tmp/browser-ui-test.version) --unsafe-perm=true && \
  python3 ../x.py check compiletest --set build.compiletest-use-stage0-libtest=true && \
  python3 ../x.py test tests/rustdoc-gui --stage 2 --test-args "'--jobs 1'"
