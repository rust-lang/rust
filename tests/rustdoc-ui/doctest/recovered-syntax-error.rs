// issue: <https://github.com/rust-lang/rust/issues/147999>
//@ compile-flags: --test
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ failure-status: 101

//! ```
//! #[derive(Clone)]
//! ```
