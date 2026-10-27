//@ check-pass
//@ compile-flags: --test --crate-type=lib

#![expect(dead_code_pub_in_binary)]
#![deny(unfulfilled_lint_expectations)]

// FIXME(#159078): `--test` causes the library crate to be treated as an
// executable, fulfilling this lint expectation incorrectly.
pub fn unused_pub_fn() {}
