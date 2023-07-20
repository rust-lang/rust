#!/bin/sh
export RUSTC_ICE=0
RUST_BACKTRACE=1 $RUSTC src/lib.rs -Z treat-err-as-bug=1 1>$TMPDIR/rust-test-1.log 2>&1
RUST_BACKTRACE=full $RUSTC src/lib.rs -Z treat-err-as-bug=1 1>$TMPDIR/rust-test-2.log 2>&1

short=$(cat $TMPDIR/rust-test-1.log | wc -l)
full=$(cat $TMPDIR/rust-test-2.log | wc -l)
rustc_query_count=$(cat $TMPDIR/rust-test-1.log | grep rustc_query_ | wc -l)
rustc_query_count_full=$(cat $TMPDIR/rust-test-2.log | grep rustc_query_ | wc -l)

begin_count=$(cat $TMPDIR/rust-test-2.log | grep __rust_begin_short_backtrace | wc -l)
end_count=$(cat $TMPDIR/rust-test-2.log | grep __rust_end_short_backtrace | wc -l)

cat $TMPDIR/rust-test-1.log
echo "====================="
cat $TMPDIR/rust-test-2.log
echo "====================="

echo "short backtrace: $short"
echo "full  backtrace: $full"
echo "begin_count: $begin_count"
echo "end_count  : $end_count"
echo "rustc_query_count: $rustc_query_count"
echo "rustc_query_count_full: $rustc_query_count_full"

## backtraces to vary a bit depending on platform and configuration options,
## here we make sure that the short backtrace of rustc_query is shorter than the full,
## and marks are in pairs.
if [ $short -lt $full ] &&
    [ $begin_count -eq $end_count ] &&
    [ $(($rustc_query_count + 5)) -lt $rustc_query_count_full ] &&
    [ $rustc_query_count_full -gt 5 ]; then
    exit 0
else
    exit 1
fi
