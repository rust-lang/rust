// Regression test for #97440: Multiline (more than 2 lines) inner attribute triggers ICE during doctest
// compile-flags:--test
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"
// normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"
// check-pass

//! ```rust
//! #![deny(
//! unused_parens,
//! )]
//! ```
