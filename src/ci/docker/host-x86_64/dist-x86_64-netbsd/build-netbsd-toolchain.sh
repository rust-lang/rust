#!/usr/bin/env bash
# ignore-tidy-linelength

set -ex

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

mkdir netbsd
cd netbsd

mkdir -p /x-tools/x86_64-unknown-netbsd/sysroot

URL=https://ci-mirrors.rust-lang.org/rustc

# Originally from ftp://ftp.netbsd.org/pub/NetBSD/NetBSD-$BSD/source/sets/*.tgz
curl $URL/2018-03-01-netbsd-src.tgz | tar xzf -
curl $URL/2018-03-01-netbsd-gnusrc.tgz | tar xzf -
curl $URL/2018-03-01-netbsd-sharesrc.tgz | tar xzf -
curl $URL/2018-03-01-netbsd-syssrc.tgz | tar xzf -

# Originally from ftp://ftp.netbsd.org/pub/NetBSD/NetBSD-$BSD/amd64/binary/sets/*.tgz
curl $URL/2018-03-01-netbsd-base.tgz | \
  tar xzf - -C /x-tools/x86_64-unknown-netbsd/sysroot ./usr/include ./usr/lib ./lib
curl $URL/2018-03-01-netbsd-comp.tgz | \
  tar xzf - -C /x-tools/x86_64-unknown-netbsd/sysroot ./usr/include ./usr/lib

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
