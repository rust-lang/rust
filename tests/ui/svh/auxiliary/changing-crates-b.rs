#![crate_name = "b"]

extern crate a;

pub fn foo() { a::foo::<isize>(); }
