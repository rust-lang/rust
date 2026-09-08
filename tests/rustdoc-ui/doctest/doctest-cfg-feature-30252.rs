//@ compile-flags:--test --cfg feature="bar"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

// https://github.com/rust-lang/rust/issues/30252
#![crate_name="issue_30252"]

/// ```rust
/// assert_eq!(cfg!(feature = "bar"), true);
/// ```
pub fn foo() {}
