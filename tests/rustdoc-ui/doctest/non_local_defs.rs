//@ check-pass
//@ compile-flags:--test --test-args --test-threads=1 --nocapture -Zunstable-options
//@ normalize-stdout-test: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stderr-test: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"

#![doc(test(attr(warn(non_local_definitions))))]

//! ```
//! #[macro_export]
//! macro_rules! a_macro { () => {} }
//! ```
