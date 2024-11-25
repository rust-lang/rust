// This checks that RUST_TEST_THREADS not being 1, 2, ... is detected
// properly.

//@ run-fail
//@ check-run-results:should be a positive integer
//@ compile-flags: --test
//@ exec-env:RUST_TEST_THREADS=foo
//@ needs-threads

#[test]
fn do_nothing() {}
