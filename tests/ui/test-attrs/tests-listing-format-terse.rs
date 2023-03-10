// no-prefer-dynamic
// compile-flags: --test
// run-flags: --list --format terse
// run-pass
// check-run-results

// Checks the listing of tests with --format terse.

#![cfg(test)]
#[test]
fn m_test() {}

#[test]
#[ignore = "not yet implemented"]
fn z_test() {}

#[test]
fn a_test() {}
