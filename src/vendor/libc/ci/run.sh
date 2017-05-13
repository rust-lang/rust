#!/bin/sh

# Builds and runs tests for a particular target passed as an argument to this
# script.

set -ex

TARGET=$1

# If we're going to run tests inside of a qemu image, then we don't need any of
# the scripts below. Instead, download the image, prepare a filesystem which has
# the current state of this repository, and then run the image.
#
# It's assume that all images, when run with two disks, will run the `run.sh`
# script from the second which we place inside.
if [ "$QEMU" != "" ]; then
  tmpdir=/tmp/qemu-img-creation
  mkdir -p $tmpdir

  if [ -z "${QEMU#*.gz}" ]; then
    # image is .gz : download and uncompress it
    qemufile=$(echo ${QEMU%.gz} | sed 's/\//__/g')
    if [ ! -f $tmpdir/$qemufile ]; then
      curl https://people.mozilla.org/~acrichton/libc-test/qemu/$QEMU | \
        gunzip -d > $tmpdir/$qemufile
    fi
  else
    # plain qcow2 image: just download it
    qemufile=$(echo ${QEMU} | sed 's/\//__/g')
    if [ ! -f $tmpdir/$qemufile ]; then
      curl https://people.mozilla.org/~acrichton/libc-test/qemu/$QEMU \
        > $tmpdir/$qemufile
    fi
  fi

  # Create a mount a fresh new filesystem image that we'll later pass to QEMU.
  # This will have a `run.sh` script will which use the artifacts inside to run
  # on the host.
  rm -f $tmpdir/libc-test.img
  mkdir $tmpdir/mount

  # If we have a cross compiler, then we just do the standard rigamarole of
  # cross-compiling an executable and then the script to run just executes the
  # binary.
  #
  # If we don't have a cross-compiler, however, then we need to do some crazy
  # acrobatics to get this to work.  Generate all.{c,rs} on the host which will
  # be compiled inside QEMU. Do this here because compiling syntex_syntax in
  # QEMU would time out basically everywhere.
  if [ "$CAN_CROSS" = "1" ]; then
    cargo build --manifest-path libc-test/Cargo.toml --target $TARGET
    cp $CARGO_TARGET_DIR/$TARGET/debug/libc-test $tmpdir/mount/
    echo 'exec $1/libc-test' > $tmpdir/mount/run.sh
  else
    rm -rf $tmpdir/generated
    mkdir -p $tmpdir/generated
    cargo build --manifest-path libc-test/generate-files/Cargo.toml
    (cd libc-test && TARGET=$TARGET OUT_DIR=$tmpdir/generated SKIP_COMPILE=1 \
      $CARGO_TARGET_DIR/debug/generate-files)

    # Copy this folder into the mounted image, the `run.sh` entry point, and
    # overwrite the standard libc-test Cargo.toml with the overlay one which will
    # assume the all.{c,rs} test files have already been generated
    mkdir $tmpdir/mount/libc
    cp -r Cargo.* libc-test src ci $tmpdir/mount/libc/
    ln -s libc-test/target $tmpdir/mount/libc/target
    cp ci/run-qemu.sh $tmpdir/mount/run.sh
    echo $TARGET | tee -a $tmpdir/mount/TARGET
    cp $tmpdir/generated/* $tmpdir/mount/libc/libc-test
    cp libc-test/run-generated-Cargo.toml $tmpdir/mount/libc/libc-test/Cargo.toml
  fi

  du -sh $tmpdir/mount
  genext2fs \
      --root $tmpdir/mount \
      --size-in-blocks 100000 \
      $tmpdir/libc-test.img

  # Pass -snapshot to prevent tampering with the disk images, this helps when
  # running this script in development. The two drives are then passed next,
  # first is the OS and second is the one we just made. Next the network is
  # configured to work (I'm not entirely sure how), and then finally we turn off
  # graphics and redirect the serial console output to out.log.
  qemu-system-x86_64 \
    -m 1024 \
    -snapshot \
    -drive if=virtio,file=$tmpdir/$qemufile \
    -drive if=virtio,file=$tmpdir/libc-test.img \
    -net nic,model=virtio \
    -net user \
    -nographic \
    -vga none 2>&1 | tee $CARGO_TARGET_DIR/out.log
  exec grep "^PASSED .* tests" $CARGO_TARGET_DIR/out.log
fi

case "$TARGET" in
  *-apple-ios)
    cargo rustc --manifest-path libc-test/Cargo.toml --target $TARGET -- \
        -C link-args=-mios-simulator-version-min=7.0
    ;;

  *)
    cargo build --manifest-path libc-test/Cargo.toml --target $TARGET
    ;;
esac

case "$TARGET" in
  arm-linux-androideabi)
    emulator @arm-21 -no-window &
    adb wait-for-device
    adb push $CARGO_TARGET_DIR/$TARGET/debug/libc-test /data/libc-test
    adb shell /data/libc-test 2>&1 | tee /tmp/out
    grep "^PASSED .* tests" /tmp/out
    ;;

  arm-unknown-linux-gnueabihf)
    qemu-arm -L /usr/arm-linux-gnueabihf $CARGO_TARGET_DIR/$TARGET/debug/libc-test
    ;;

  mips-unknown-linux-gnu)
    qemu-mips -L /usr/mips-linux-gnu $CARGO_TARGET_DIR/$TARGET/debug/libc-test
    ;;

  mips64-unknown-linux-gnuabi64)
    qemu-mips64 -L /usr/mips64-linux-gnuabi64 $CARGO_TARGET_DIR/$TARGET/debug/libc-test
    ;;

  mips-unknown-linux-musl)
    qemu-mips -L /toolchain/staging_dir/toolchain-mips_34kc_gcc-5.3.0_musl-1.1.15 \
              $CARGO_TARGET_DIR/$TARGET/debug/libc-test
    ;;

  mipsel-unknown-linux-musl)
      qemu-mipsel -L /toolchain $CARGO_TARGET_DIR/$TARGET/debug/libc-test
      ;;

  powerpc-unknown-linux-gnu)
    qemu-ppc -L /usr/powerpc-linux-gnu $CARGO_TARGET_DIR/$TARGET/debug/libc-test
    ;;

  powerpc64-unknown-linux-gnu)
    qemu-ppc64 -L /usr/powerpc64-linux-gnu $CARGO_TARGET_DIR/$TARGET/debug/libc-test
    ;;

  aarch64-unknown-linux-gnu)
    qemu-aarch64 -L /usr/aarch64-linux-gnu/ $CARGO_TARGET_DIR/$TARGET/debug/libc-test
    ;;

  *-rumprun-netbsd)
    rumprun-bake hw_virtio /tmp/libc-test.img $CARGO_TARGET_DIR/$TARGET/debug/libc-test
    qemu-system-x86_64 -nographic -vga none -m 64 \
        -kernel /tmp/libc-test.img 2>&1 | tee /tmp/out &
    sleep 5
    grep "^PASSED .* tests" /tmp/out
    ;;

  *)
    $CARGO_TARGET_DIR/$TARGET/debug/libc-test
    ;;
esac
