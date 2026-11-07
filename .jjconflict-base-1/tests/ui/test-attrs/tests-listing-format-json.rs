//@ no-prefer-dynamic
//@ compile-flags: --test
//@ run-flags: --list --format json -Zunstable-options
//@ run-pass
//@ check-run-results
//@ only-nightly
//@ normalize-stdout: "fake-test-src-base/test-attrs/" -> "$$DIR/"
//@ normalize-stdout: "fake-test-src-base\\test-attrs\\" -> "$$DIR/"

// Checks the listing of tests with --format json.

#![cfg(test)]
#[test]
fn m_test() {}

#[test]
#[ignore = "not yet implemented"]
fn z_test() {}

#[test]
fn a_test() {}
