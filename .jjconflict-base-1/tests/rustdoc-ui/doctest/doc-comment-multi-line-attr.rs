// Regression test for #97440: Multiline inner attribute triggers ICE during doctest
//@ compile-flags:--test
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

//! ```rust
//! #![deny(
//! unused_parens,
//! )]
//! ```
