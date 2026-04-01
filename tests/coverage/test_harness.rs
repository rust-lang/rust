// Verify that the entry point injected by the test harness doesn't cause
// weird artifacts in the coverage report (e.g. issue #10749).

//@ compile-flags: --test

#[allow(dead_code)]
fn unused() {}

#[test]
fn my_test() {}
