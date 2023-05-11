// Test that `include!` in a doc test searches relative to the directory in
// which the test is declared.

// compile-flags:--test

/// ```rust
/// include!("auxiliary/empty.rs");
/// fn main() {}
/// ```
pub struct Foo;
