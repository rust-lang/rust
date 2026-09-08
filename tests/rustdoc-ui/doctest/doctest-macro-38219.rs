// https://github.com/rust-lang/rust/issues/38219

//@ compile-flags:--test
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ should-fail

/// ```
/// fail
/// ```
#[macro_export]
macro_rules! foo { () => {} }
