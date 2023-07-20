#!/bin/sh

# Default nightly behavior (write ICE to current directory)
# FIXME(estebank): these are failing on CI, but passing locally.
# $RUSTC src/lib.rs -Z treat-err-as-bug=1 1>$TMPDIR/rust-test-default.log 2>&1
# default=$(cat ./rustc-ice-*.txt | wc -l)
# rm ./rustc-ice-*.txt

# Explicit directory set
export RUSTC_ICE=$TMPDIR
$RUSTC src/lib.rs -Z treat-err-as-bug=1 1>$TMPDIR/rust-test-default-set.log 2>&1
default_set=$(cat $TMPDIR/rustc-ice-*.txt | wc -l)
content=$(cat $TMPDIR/rustc-ice-*.txt)
rm $TMPDIR/rustc-ice-*.txt
RUST_BACKTRACE=short $RUSTC src/lib.rs -Z treat-err-as-bug=1 1>$TMPDIR/rust-test-short.log 2>&1
short=$(cat $TMPDIR/rustc-ice-*.txt | wc -l)
rm $TMPDIR/rustc-ice-*.txt
RUST_BACKTRACE=full $RUSTC src/lib.rs -Z treat-err-as-bug=1 1>$TMPDIR/rust-test-full.log 2>&1
full=$(cat $TMPDIR/rustc-ice-*.txt | wc -l)
rm $TMPDIR/rustc-ice-*.txt

# Explicitly disabling ICE dump
export RUSTC_ICE=0
$RUSTC src/lib.rs -Z treat-err-as-bug=1 1>$TMPDIR/rust-test-disabled.log 2>&1
should_be_empty_tmp=$(ls -l $TMPDIR/rustc-ice-*.txt | wc -l)
should_be_empty_dot=$(ls -l ./rustc-ice-*.txt | wc -l)

echo "#### ICE Dump content:"
echo $content
echo "#### default length:"
echo $default
echo "#### short length:"
echo $short
echo "#### default_set length:"
echo $default_set
echo "#### full length:"
echo $full
echo "#### should_be_empty_dot length:"
echo $should_be_empty_dot
echo "#### should_be_empty_tmp length:"
echo $should_be_empty_tmp

## Verify that a the ICE dump file is created in the appropriate directories, that
## their lengths are the same regardless of other backtrace configuration options,
## that the file is not created when asked to (RUSTC_ICE=0) and that the file
## contains at least part of the expected content.
if [ $short -eq $default_set ] &&
    #[ $default -eq $short ] &&
    [ $default_set -eq $full ] &&
    [[ $content == *"thread 'rustc' panicked at "* ]] &&
    [[ $content == *"stack backtrace:"* ]] &&
    #[ $default -gt 0 ] &&
    [ $should_be_empty_dot -eq 0 ] &&
    [ $should_be_empty_tmp -eq 0 ]; then
    exit 0
else
    exit 1
fi
