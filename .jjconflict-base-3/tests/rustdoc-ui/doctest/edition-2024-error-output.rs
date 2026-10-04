// This is a regression test for <https://github.com/rust-lang/rust/issues/137970>.
// The output must look nice and not like a `Debug` display of a `String`.

//@ edition: 2024
//@ compile-flags: --test
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "panicked at .+rs:" -> "panicked at $$TMP:"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout: "ran in \d+\.\d+s" -> "ran in $$TIME"
//@ normalize-stdout: "compilation took \d+\.\d+s" -> "compilation took $$TIME"
//@ rustc-env:RUST_BACKTRACE=0
//@ failure-status: 101

//! ```rust
//! assert_eq!(2 + 2, 5);
//! ```
