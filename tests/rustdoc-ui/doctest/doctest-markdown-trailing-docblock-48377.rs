//@ compile-flags:--test --test-args=--test-threads=1
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

// https://github.com/rust-lang/rust/issues/48377

//! This is a doc comment
//!
//! ```rust
//! fn main() {}
//! ```
//!
//! With a trailing code fence
//! ```

/// Some foo function
pub fn foo() {}
