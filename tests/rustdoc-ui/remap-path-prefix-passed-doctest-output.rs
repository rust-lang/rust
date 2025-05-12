//@ check-pass
//@ check-run-results

// FIXME: if/when the output of the test harness can be tested on its own, this test should be
// adapted to use that, and that normalize line can go away

//@ compile-flags:--test -Z unstable-options --remap-path-prefix={{src-base}}=remapped_path --test-args --test-threads=1
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"

// doctest passes at runtime
/// ```
/// assert!(true);
/// ```
pub struct SomeStruct;
