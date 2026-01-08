//@ compile-flags:--test --cfg feature="bar"

// https://github.com/rust-lang/rust/issues/30252
#![crate_name="issue_30252"]

/// ```rust
/// assert_eq!(cfg!(feature = "bar"), true);
/// ```
pub fn foo() {}
