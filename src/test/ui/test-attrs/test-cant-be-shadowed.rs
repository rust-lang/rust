// build-pass (FIXME(62277): could be check-pass?)
// aux-build:test_macro.rs
// compile-flags:--test

#[macro_use] extern crate test_macro;

#[test]
fn foo(){}

macro_rules! test { () => () }

#[test]
fn bar() {}
