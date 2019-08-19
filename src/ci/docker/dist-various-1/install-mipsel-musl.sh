set -ex

mkdir /usr/local/mipsel-linux-musl

# Note that this originally came from:
# https://downloads.openwrt.org/snapshots/trunk/malta/generic/
# OpenWrt-Toolchain-malta-le_gcc-5.3.0_musl-1.1.15.Linux-x86_64.tar.bz2
URL="https://rust-lang-ci2.s3.amazonaws.com/libc"
FILE="OpenWrt-Toolchain-malta-le_gcc-5.3.0_musl-1.1.15.Linux-x86_64.tar.bz2"
curl -L "$URL/$FILE" | tar xjf - -C /usr/local/mipsel-linux-musl --strip-components=2

for file in /usr/local/mipsel-linux-musl/bin/mipsel-openwrt-linux-*; do
  ln -s $file /usr/local/bin/`basename $file`
done
