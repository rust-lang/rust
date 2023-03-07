// compile-flags:--test --cfg feature="bar"

/// ```rust
/// assert_eq!(cfg!(feature = "bar"), true);
/// ```
pub fn foo() {}
