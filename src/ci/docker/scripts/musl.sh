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

MUSL=musl-1.2.3

# may have been downloaded in a previous run
if [ ! -d $MUSL ]; then
  curl https://www.musl-libc.org/releases/$MUSL.tar.gz | tar xzf -

  # Apply patches for CVE-2025-26519. At the time of adding these patches no release containing them
  # has been published by the musl project, so we just apply them directly on top of the version we
  # were distributing already. The patches should be removed once we upgrade to musl >= 1.2.6.
  #
  # Advisory: https://www.openwall.com/lists/musl/2025/02/13/1
  #
  # Patches applied:
  # - https://www.openwall.com/lists/musl/2025/02/13/1/1
  # - https://www.openwall.com/lists/musl/2025/02/13/1/2
  #
  # ignore-tidy-tab
  # ignore-tidy-linelength
  patch -p1 -d $MUSL <<EOF
--- a/src/locale/iconv.c
+++ b/src/locale/iconv.c
@@ -502,7 +502,7 @@ size_t iconv(iconv_t cd, char **restrict in, size_t *restrict inb, char **restri
 			if (c >= 93 || d >= 94) {
 				c += (0xa1-0x81);
 				d += 0xa1;
-				if (c >= 93 || c>=0xc6-0x81 && d>0x52)
+				if (c > 0xc6-0x81 || c==0xc6-0x81 && d>0x52)
 					goto ilseq;
 				if (d-'A'<26) d = d-'A';
 				else if (d-'a'<26) d = d-'a'+26;
EOF
  patch -p1 -d $MUSL <<EOF
--- a/src/locale/iconv.c
+++ b/src/locale/iconv.c
@@ -545,6 +545,10 @@ size_t iconv(iconv_t cd, char **restrict in, size_t *restrict inb, char **restri
 				if (*outb < k) goto toobig;
 				memcpy(*out, tmp, k);
 			} else k = wctomb_utf8(*out, c);
+			/* This failure condition should be unreachable, but
+			 * is included to prevent decoder bugs from translating
+			 * into advancement outside the output buffer range. */
+			if (k>4) goto ilseq;
 			*out += k;
 			*outb -= k;
 			break;
EOF
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
