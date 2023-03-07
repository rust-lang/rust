// no-prefer-dynamic
// compile-flags: --test
// run-flags: --test-threads=1
// run-pass
// check-run-results
// only-wasm32

// Tests the output of the test harness with only passed tests.

#![cfg(test)]

#[test]
fn it_works() {
    assert_eq!(1 + 1, 2);
}

#[test]
fn it_works_too() {
    assert_eq!(1 * 0, 0);
}
