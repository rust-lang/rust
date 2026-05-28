//@ proc-macro: test-macros.rs

#[macro_use]
extern crate test_macros;
#[macro_use]
extern crate test_macros; //~ ERROR the name `test_macros` is defined multiple times

fn main() {}
