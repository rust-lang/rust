set -ex

apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates \
  cmake \
  curl \
  file \
  g++ \
  git \
  libssl-dev \
  make \
  pkg-config \
  python3 \
  sudo \
  unzip \
  xz-utils
