// FIXME: if/when the output of the test harness can be tested on its own, this test should be
// adapted to use that, and that normalize line can go away

//@ failure-status: 101
//@ compile-flags:--test -Z unstable-options --remap-path-prefix={{src-base}}=remapped_path --test-args --test-threads=1
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout: "exit (status|code): 101" -> "exit status: 101"

// doctest fails at runtime
/// ```
/// panic!("oh no");
/// ```
pub struct SomeStruct;
