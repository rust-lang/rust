// compile-flags:--test --cfg feature="bar"

#![crate_name="issue_30252"]

/// ```rust
/// assert_eq!(cfg!(feature = "bar"), true);
/// ```
pub fn foo() {}
