//@ proc-macro: test-macros.rs

extern crate test_macros;

#[test_macros::panic_attr] //~ ERROR custom attribute panicked
fn foo() {}

fn main() {}
