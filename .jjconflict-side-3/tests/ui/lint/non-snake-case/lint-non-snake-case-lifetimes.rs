#![deny(non_snake_case)]
#![allow(dead_code)]

fn f<'FooBar>( //~ ERROR lifetime `'FooBar` should have a snake case name
    _: &'FooBar ()
) {}

fn main() { }
