//@ compile-flags: --test

// This is how cargo compiles a library's unit tests: `--test` and no `--crate-type`.
// It must not be assumed to be a binary (#159078).
//
// FIXME: this is also how a binary's tests would look if `rustc --test` is invoked
// by hand, so that case is a known false negative. Cargo currently omits
// `--crate-type` for these test targets. A possible follow-up is for Cargo
// to pass `--crate-type bin` for binary-like targets.
#![expect(dead_code_pub_in_binary)] //~ ERROR this lint expectation is unfulfilled
#![deny(unfulfilled_lint_expectations)]

pub fn unused_pub_fn() {}
