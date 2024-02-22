//@ compile-flags:--test

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
