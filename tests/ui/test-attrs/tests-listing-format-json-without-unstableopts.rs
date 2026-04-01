//@ no-prefer-dynamic
//@ compile-flags: --test
//@ run-flags: --list --format json
//@ run-fail
//@ check-run-results

// Checks that --format json does not work without -Zunstable-options.

#![cfg(test)]
#[test]
fn m_test() {}

#[test]
#[ignore = "not yet implemented"]
fn z_test() {}

#[test]
fn a_test() {}
