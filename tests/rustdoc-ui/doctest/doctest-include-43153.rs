// https://github.com/rust-lang/rust/issues/43153

// Test that `include!` in a doc test searches relative to the directory in
// which the test is declared.

//@ compile-flags:--test
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

/// ```rust
/// include!("auxiliary/empty.rs");
/// fn main() {}
/// ```
pub struct Foo;
