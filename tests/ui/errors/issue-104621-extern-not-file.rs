//@ compile-flags: --extern foo=.

extern crate foo; //~ ERROR extern location for foo is not a file: .
fn main() {}
