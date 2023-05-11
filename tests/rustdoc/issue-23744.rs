// compile-flags:--test

/// Example of rustdoc incorrectly parsing <code>```rust,should_panic</code>.
///
/// ```should_panic
/// fn main() { panic!("fee"); }
/// ```
///
/// ```rust,should_panic
/// fn main() { panic!("fum"); }
/// ```
pub fn foo() {}
