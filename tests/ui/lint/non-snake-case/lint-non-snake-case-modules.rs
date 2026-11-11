#![deny(non_snake_case)]
#![allow(dead_code)]

mod FooBar { //~ ERROR module `FooBar` should have a snake case name
    pub struct S;
}

fn f(_: FooBar::S) { }

fn main() { }
