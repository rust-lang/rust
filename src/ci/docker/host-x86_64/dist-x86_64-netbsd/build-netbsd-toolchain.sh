#!/usr/bin/env bash
# ignore-tidy-linelength

set -eux

hide_output() {
  set +x
  on_err="
echo ERROR: An error was encountered with the build.
cat /tmp/build.log
exit 1
"
  trap "$on_err" ERR
  bash -c "while true; do sleep 30; echo \$(date) - building ...; done" &
  PING_LOOP_PID=$!
  "$@" &> /tmp/build.log
  rm /tmp/build.log
  trap - ERR
  kill $PING_LOOP_PID
  set -x
}

# Download, verify SHA512, and remove the downloaded file
# Usage: <file name> <url> <file sha> <full tar command using fname>
download() {
  fname="$1"
  shift
  url="$1"
  shift
  sha="$1"
  shift

  curl "$url" -o "$fname"
  echo "$sha  $fname" | shasum -a 512 --check || exit 1
  "$@"
  rm "$fname"
}

mkdir netbsd
cd netbsd

mkdir -p /x-tools/x86_64-unknown-netbsd/sysroot

# Hashes come from https://cdn.netbsd.org/pub/NetBSD/security/hashes/NetBSD-9.0_hashes.asc
SRC_SHA=2c791ae009a6929c6fc893ec5df7e62910ee8207e0b2159d6937309c03efe175b6ae1e445829a13d041b6851334ad35c521f2fa03c97675d4a05f1fafe58ede0
GNUSRC_SHA=3710085a73feecf6a843415271ec794c90146b03f6bbd30f07c9e0c79febf8995d557e40194f1e05db655e4f5ef2fae97563f8456fceaae65d4ea98857a83b1c
SHARESRC_SHA=f080776ed82c3ac5d6272dee39746f87897d8e6984996caf5bf6d87bf11d9c9e0c1ad5c437c21258bd278bb6fd76974946e878f548517885f71c556096231369
SYSSRC_SHA=60b9ddf4cc6402256473e2e1eefeabd9001aa4e205208715ecc6d6fc3f5b400e469944580077271b8e80562a4c2f601249e69e07a504f46744e0c50335f1cbf1
BASE_SHA=b5926b107cebf40c3c19b4f6cd039b610987dd7f819e7cdde3bd1e5230a856906e7930b15ab242d52ced9f0bda01d574be59488b8dbb95fa5df2987d0a70995f
COMP_SHA=38ea54f30d5fc2afea87e5096f06873e00182789e8ad9cec0cb3e9f7c538c1aa4779e63fd401a36ba02676158e83fa5c95e8e87898db59c1914fb206aecd82d2

SOURCE_URL=https://ci-mirrors.rust-lang.org/rustc/2025-03-14-netbsd-9.0-src
download src.tgz "$SOURCE_URL-src.tgz" "$SRC_SHA" tar xzf src.tgz
download gnusrc.tgz "$SOURCE_URL-gnusrc.tgz" "$GNUSRC_SHA" tar xzf gnusrc.tgz
download sharesrc.tgz "$SOURCE_URL-sharesrc.tgz" "$SHARESRC_SHA" tar xzf sharesrc.tgz
download syssrc.tgz "$SOURCE_URL-syssrc.tgz" "$SYSSRC_SHA" tar xzf syssrc.tgz

BINARY_URL=https://ci-mirrors.rust-lang.org/rustc/2025-03-14-netbsd-9.0-amd64-binary
download base.tar.xz "$BINARY_URL-base.tar.xz" "$BASE_SHA" \
  tar xJf base.tar.xz -C /x-tools/x86_64-unknown-netbsd/sysroot ./usr/include ./usr/lib ./lib
download comp.tar.xz "$BINARY_URL-comp.tar.xz" "$COMP_SHA" \
  tar xJf comp.tar.xz -C /x-tools/x86_64-unknown-netbsd/sysroot ./usr/include ./usr/lib

cd usr/src

# The options, in order, do the following
# * this is an unprivileged build
# * output to a predictable location
# * disable various unneeded stuff
MKUNPRIVED=yes TOOLDIR=/x-tools/x86_64-unknown-netbsd \
MKSHARE=no MKDOC=no MKHTML=no MKINFO=no MKKMOD=no MKLINT=no MKMAN=no MKNLS=no MKPROFILE=no \
hide_output ./build.sh -j10 -m amd64 tools

cd ../..

rm -rf usr

cat > /x-tools/x86_64-unknown-netbsd/bin/x86_64--netbsd-gcc-sysroot <<'EOF'
#!/usr/bin/env bash
exec /x-tools/x86_64-unknown-netbsd/bin/x86_64--netbsd-gcc --sysroot=/x-tools/x86_64-unknown-netbsd/sysroot "$@"
EOF

cat > /x-tools/x86_64-unknown-netbsd/bin/x86_64--netbsd-g++-sysroot <<'EOF'
#!/usr/bin/env bash
exec /x-tools/x86_64-unknown-netbsd/bin/x86_64--netbsd-g++ --sysroot=/x-tools/x86_64-unknown-netbsd/sysroot "$@"
EOF

GCC_SHA1=`sha1sum -b /x-tools/x86_64-unknown-netbsd/bin/x86_64--netbsd-gcc | cut -d' ' -f1`
GPP_SHA1=`sha1sum -b /x-tools/x86_64-unknown-netbsd/bin/x86_64--netbsd-g++ | cut -d' ' -f1`

echo "# $GCC_SHA1" >> /x-tools/x86_64-unknown-netbsd/bin/x86_64--netbsd-gcc-sysroot
echo "# $GPP_SHA1" >> /x-tools/x86_64-unknown-netbsd/bin/x86_64--netbsd-g++-sysroot

chmod +x /x-tools/x86_64-unknown-netbsd/bin/x86_64--netbsd-gcc-sysroot
chmod +x /x-tools/x86_64-unknown-netbsd/bin/x86_64--netbsd-g++-sysroot
