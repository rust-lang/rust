// run-pass
// Check that `#[test]` works with extern-absolute-paths enabled.
//
// Regression test for #47075.

// edition:2018
// compile-flags: --test

#[test]
fn test() {}
