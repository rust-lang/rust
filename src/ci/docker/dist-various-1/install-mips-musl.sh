set -ex

mkdir /usr/local/mips-linux-musl

# originally from
# https://downloads.openwrt.org/snapshots/trunk/ar71xx/generic/
# OpenWrt-Toolchain-ar71xx-generic_gcc-5.3.0_musl-1.1.16.Linux-x86_64.tar.bz2
URL="https://rust-lang-ci2.s3.amazonaws.com/rust-ci-mirror"
FILE="OpenWrt-Toolchain-ar71xx-generic_gcc-5.3.0_musl-1.1.16.Linux-x86_64.tar.bz2"
curl -L "$URL/$FILE" | tar xjf - -C /usr/local/mips-linux-musl --strip-components=2

for file in /usr/local/mips-linux-musl/bin/mips-openwrt-linux-*; do
  ln -s $file /usr/local/bin/`basename $file`
done
