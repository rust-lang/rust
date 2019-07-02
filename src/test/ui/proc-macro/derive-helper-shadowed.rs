// build-pass (FIXME(62277): could be check-pass?)
// aux-build:test-macros.rs
// aux-build:derive-helper-shadowed-2.rs

#[macro_use]
extern crate test_macros;
#[macro_use(empty_helper)]
extern crate derive_helper_shadowed_2;

macro_rules! empty_helper { () => () }

#[derive(Empty)]
#[empty_helper] // OK
struct S;

fn main() {}
