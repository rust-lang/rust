#!/usr/bin/env bash
# ignore-tidy-file-linelength

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

# Hashes come from https://cdn.netbsd.org/pub/NetBSD/security/hashes/NetBSD-10.1_hashes.asc
SRC_SHA=6ae2053b4b75821238c0757d4f7258daece425de72524c616e07d3adee7c48d87422dd47d852a137918cec3dd3c0d339e372f4504dfe9f1bc5520011775bdb86
GNUSRC_SHA=8a1c42030ba44eb2a0c7a5111187bc02e8f4d0860d8491b7863579e612333665c478625c37b01f08732e3cfd29ec31335f1db1274fd7dcfdc048b09d1b4bbb83
SHARESRC_SHA=703eeb306fc0328cad7e6f0e100d2e7af194f82e613338f4611a7bcd5f6d773d8789e7ce03ec25268ec2b95ccdb97c3b4289a838a629716498b4d7c3184cb1ef
SYSSRC_SHA=766ac21f33cfe0e701dfedb894fa07f36d811da1a12e979181e8fca7af4e627852680ce42a7b29e97dd3e2e402ddf9ae7bfba60c8d7dc6b8a3354d8ce8c06926
BASE_SHA=1c6b74cedd7fc92adfe51239808d41e63fc6eb9e9877c1101c1da515a38457278c7774e8fc3041d69cefc49821bad836cfdcc5147ad757f2f4a1f59f9e3862e5
COMP_SHA=ea90d527d1475bd64d2b4c36d9d6ae53cb353cbc8c4feaa2b2bf006710282d2f47349a8aeeb64531f9da04268ed368ff2603c1e11a30bdfc8b2539ad73fe725f

SOURCE_URL=https://ci-mirrors.rust-lang.org/rustc/netbsd/2026-07-28-netbsd-10.1-src
download src.tgz "$SOURCE_URL-src.tgz" "$SRC_SHA" tar xzf src.tgz
download gnusrc.tgz "$SOURCE_URL-gnusrc.tgz" "$GNUSRC_SHA" tar xzf gnusrc.tgz
download sharesrc.tgz "$SOURCE_URL-sharesrc.tgz" "$SHARESRC_SHA" tar xzf sharesrc.tgz
download syssrc.tgz "$SOURCE_URL-syssrc.tgz" "$SYSSRC_SHA" tar xzf syssrc.tgz

BINARY_URL=https://ci-mirrors.rust-lang.org/rustc/netbsd/2026-07-28-netbsd-10.1-amd64-binary
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
