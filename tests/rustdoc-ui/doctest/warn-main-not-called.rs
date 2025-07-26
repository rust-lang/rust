//@ check-pass
//@ compile-flags:--test --test-args --test-threads=1
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"

// In case there is a `main` function in the doctest alongside expressions,
// the whole doctest will be wrapped into a function and the `main` function
// won't be called.

//! ```
//! macro_rules! bla {
//!     ($($x:tt)*) => {}
//! }
//!
//! let x = 12;
//! bla!(fn main ());
//! ```
//!
//! ```
//! let x = 12;
//! fn main() {}
//! ```
