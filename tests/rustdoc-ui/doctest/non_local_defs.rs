//@ check-pass
//@ compile-flags:--test --test-args --test-threads=1 --nocapture -Zunstable-options
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stderr: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"

//! ```
//! #[macro_export]
//! macro_rules! a_macro { () => {} }
//! ```
