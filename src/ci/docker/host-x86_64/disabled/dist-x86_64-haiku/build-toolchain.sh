#!/usr/bin/env bash

set -ex

ARCH=$1

TOP=$(pwd)

BUILDTOOLS=$TOP/buildtools
HAIKU=$TOP/haiku
OUTPUT=/tools
SYSROOT=$OUTPUT/cross-tools-$ARCH/sysroot
PACKAGE_ROOT=/system

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
  set -x
}

# First up, build a cross-compiler
git clone --depth=1 https://git.haiku-os.org/haiku
git clone --depth=1 https://git.haiku-os.org/buildtools
cd $BUILDTOOLS/jam
hide_output make
hide_output ./jam0 install
mkdir -p $OUTPUT
cd $OUTPUT
hide_output $HAIKU/configure --build-cross-tools $ARCH $TOP/buildtools

# Set up sysroot to redirect to /system
mkdir -p $SYSROOT/boot
mkdir -p $PACKAGE_ROOT
ln -s $PACKAGE_ROOT $SYSROOT/boot/system

# Build needed packages and tools for the cross-compiler
hide_output jam -q haiku.hpkg haiku_devel.hpkg '<build>package'

# Set up our sysroot
cp $OUTPUT/objects/linux/lib/*.so /lib/x86_64-linux-gnu
cp $OUTPUT/objects/linux/x86_64/release/tools/package/package /bin/
find $SYSROOT/../bin/ -type f -exec ln -s {} /bin/ \;

# Extract packages
package extract -C $PACKAGE_ROOT $OUTPUT/objects/haiku/$ARCH/packaging/packages/haiku.hpkg
package extract -C $PACKAGE_ROOT $OUTPUT/objects/haiku/$ARCH/packaging/packages/haiku_devel.hpkg
find $OUTPUT/download/ -name '*.hpkg' -exec package extract -C $PACKAGE_ROOT {} \;

# Fix libgcc_s so we can link to it
cd $PACKAGE_ROOT/develop/lib
ln -s ../../lib/libgcc_s.so libgcc_s.so

# Clean up
rm -rf $BUILDTOOLS $HAIKU $OUTPUT/Jamfile $OUTPUT/attributes $OUTPUT/build \
  $OUTPUT/build_packages $OUTPUT/download $OUTPUT/objects
