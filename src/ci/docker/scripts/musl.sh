#!/bin/sh
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
  trap - ERR
  kill $PING_LOOP_PID
  rm /tmp/build.log
  set -x
}

TAG=$1
shift

# Ancient binutils versions don't understand debug symbols produced by more recent tools.
# Apparently applying `-fPIC` everywhere allows them to link successfully.
export CFLAGS="-fPIC $CFLAGS"

MUSL=musl-1.2.6

# may have been downloaded in a previous run
if [ ! -d $MUSL ]; then
  curl https://www.musl-libc.org/releases/$MUSL.tar.gz | tar xzf -

  # Apply patches for CVE-2026-6042, CVE-2026-40200 and CVE-2026-107891.
  #
  # At the time of adding these patches no release containing them has been published by the musl
  # project, so we just apply them directly on top of the version we were distributing already. The
  # patches should be removed once we upgrade to musl >= 1.2.7.
  #
  SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
  MUSL_PATCH_DIR="$SCRIPT_DIR"
  # Advisory: https://www.openwall.com/lists/oss-security/2026/04/09/19
  # Patches:  https://www.openwall.com/lists/musl/2026/04/03/2/1
  patch -p1 -d $MUSL <"$MUSL_PATCH_DIR/musl-cve-2026-6042.diff"
  # Advisory: https://www.openwall.com/lists/musl/2026/04/10/3
  # Patches:  https://www.openwall.com/lists/musl/2026/04/10/3/1
  patch -p1 -d $MUSL <"$MUSL_PATCH_DIR/musl-cve-2026-40200.diff"
  # Advisory: https://www.openwall.com/lists/musl/2026/10/21/1
  # Patch:    https://www.openwall.com/lists/musl/2026/10/21/1/1
  patch -p1 -d $MUSL <"$MUSL_PATCH_DIR/musl-cve-2026-107891.diff"
fi

cd $MUSL
./configure --enable-debug --disable-shared --prefix=/musl-$TAG "$@"
if [ "$TAG" = "i586" -o "$TAG" = "i686" ]; then
  hide_output make -j$(nproc) AR=ar RANLIB=ranlib
else
  hide_output make -j$(nproc)
fi
hide_output make install
hide_output make clean
