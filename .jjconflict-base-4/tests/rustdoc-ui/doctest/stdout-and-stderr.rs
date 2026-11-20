// This test ensures that the output is correctly generated when the
// doctest fails. It checks when there is stderr and stdout, no stdout
// and no stderr/stdout.
//
// This is a regression test for <https://github.com/rust-lang/rust/issues/140289>.

//@ edition: 2024
//@ compile-flags:--test --test-args=--test-threads=1
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout: "panicked at .+rs:" -> "panicked at $$TMP:"
//@ normalize-stdout: "ran in \d+\.\d+s" -> "ran in $$TIME"
//@ normalize-stdout: "compilation took \d+\.\d+s" -> "compilation took $$TIME"
//@ failure-status: 101
//@ rustc-env:RUST_BACKTRACE=0

//! ```
//! println!("######## from a DOC TEST ########");
//! assert_eq!("doc", "test");
//! ```
//!
//! ```
//! assert_eq!("doc", "test");
//! ```
//!
//! ```
//! std::process::exit(1);
//! ```
