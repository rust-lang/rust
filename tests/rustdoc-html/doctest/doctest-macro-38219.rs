// https://github.com/rust-lang/rust/issues/38219

//@ compile-flags:--test
//@ should-fail

/// ```
/// fail
/// ```
#[macro_export]
macro_rules! foo { () => {} }
