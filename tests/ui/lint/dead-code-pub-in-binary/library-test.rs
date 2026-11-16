//@ check-pass
//@ compile-flags: --test --crate-type=lib

#![expect(dead_code_pub_in_binary)]
#![deny(unfulfilled_lint_expectations)]

// FIXME(#159078): `--test` causes rustc to treat this library as an
// executable, incorrectly fulfilling the lint expectation. Cargo works
// around this for library test units (rust-lang/cargo#17608).
pub fn unused_pub_fn() {}
