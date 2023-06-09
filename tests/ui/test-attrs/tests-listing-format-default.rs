// no-prefer-dynamic
// compile-flags: --test
// run-flags: --list
// run-pass
// check-run-results

// Checks the listing of tests with no --format arguments.

#![cfg(test)]
#[test]
fn m_test() {}

#[test]
#[ignore = "not yet implemented"]
fn z_test() {}

#[test]
fn a_test() {}
