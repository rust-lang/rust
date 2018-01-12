# Copyright 2016 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

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
  $@ &> /tmp/build.log
  trap - ERR
  kill $PING_LOOP_PID
  rm /tmp/build.log
  set -x
}

TAG=$1
shift

MUSL=musl-1.1.18

# may have been downloaded in a previous run
if [ ! -d $MUSL ]; then
  curl https://www.musl-libc.org/releases/$MUSL.tar.gz | tar xzf -
fi

cd $MUSL
./configure --disable-shared --prefix=/musl-$TAG $@
if [ "$TAG" = "i586" -o "$TAG" = "i686" ]; then
  hide_output make -j$(nproc) AR=ar RANLIB=ranlib
else
  hide_output make -j$(nproc)
fi
hide_output make install
hide_output make clean

cd ..

LLVM=39
# may have been downloaded in a previous run
if [ ! -d libunwind-release_$LLVM ]; then
  curl -L https://github.com/llvm-mirror/llvm/archive/release_$LLVM.tar.gz | tar xzf -
  curl -L https://github.com/llvm-mirror/libunwind/archive/release_$LLVM.tar.gz | tar xzf -
  # Whoa what's this mysterious patch we're applying to libunwind! Why are we
  # swapping the values of ESP/EBP in libunwind?!
  #
  # Discovered in #35599 it turns out that the vanilla build of libunwind is not
  # suitable for unwinding i686 musl. After some investigation it ended up
  # looking like the register values for ESP/EBP were indeed incorrect (swapped)
  # in the source. Similar commits in libunwind (r280099 and r282589) have noticed
  # this for other platforms, and we just need to realize it for musl linux as
  # well.
  #
  # More technical info can be found at #35599
  cd libunwind-release_$LLVM
  patch -Np1 << EOF
diff --git a/include/libunwind.h b/include/libunwind.h
index c5b9633..1360eb2 100644
--- a/include/libunwind.h
+++ b/include/libunwind.h
@@ -151,8 +151,8 @@ enum {
   UNW_X86_ECX = 1,
   UNW_X86_EDX = 2,
   UNW_X86_EBX = 3,
-  UNW_X86_EBP = 4,
-  UNW_X86_ESP = 5,
+  UNW_X86_ESP = 4,
+  UNW_X86_EBP = 5,
   UNW_X86_ESI = 6,
   UNW_X86_EDI = 7
 };
fi
EOF
  cd ..
fi

mkdir libunwind-build
cd libunwind-build
cmake ../libunwind-release_$LLVM \
          -DLLVM_PATH=/build/llvm-release_$LLVM \
          -DLIBUNWIND_ENABLE_SHARED=0 \
          -DCMAKE_C_COMPILER=$CC \
          -DCMAKE_CXX_COMPILER=$CXX \
          -DCMAKE_C_FLAGS="$CFLAGS" \
          -DCMAKE_CXX_FLAGS="$CXXFLAGS"

hide_output make -j$(nproc)
cp lib/libunwind.a /musl-$TAG/lib
cd ../ && rm -rf libunwind-build
