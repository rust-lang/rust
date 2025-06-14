//@ check-pass
//@ compile-flags:--test --test-arg --test-threads=1 --test-arg --nocapture -Zunstable-options
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stderr: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"

//! ```
//! #[macro_export]
//! macro_rules! a_macro { () => {} }
//! ```
