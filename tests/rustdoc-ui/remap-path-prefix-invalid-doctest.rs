// FIXME: if/when the output of the test harness can be tested on its own, this test should be
// adapted to use that, and that normalize line can go away

//@ failure-status: 101
//@ compile-flags:--test -Z unstable-options --remap-path-prefix={{src-base}}=remapped_path --test-args --test-threads=1
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"

// doctest fails to compile
/// ```
/// this is not real code
/// ```
pub struct SomeStruct;
