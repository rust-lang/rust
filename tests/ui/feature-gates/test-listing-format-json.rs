//@ no-prefer-dynamic
//@ compile-flags: --test
//@ run-flags: --list --format json -Zunstable-options
//@ run-fail
//@ check-run-results
//@ ignore-nightly
//@ unset-exec-env:RUSTC_BOOTSTRAP

#![cfg(test)]
#[test]
fn m_test() {}

#[test]
#[ignore = "not yet implemented"]
fn z_test() {}

#[test]
fn a_test() {}
