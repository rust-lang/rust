// https://github.com/rust-lang/rust/issues/38219

//@ compile-flags:--test
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ failure-status: 101

/// ```
/// fail
/// ```
#[macro_export]
macro_rules! foo { () => {} }
