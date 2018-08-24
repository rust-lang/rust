#![deny(non_snake_case)]
#![allow(dead_code)]

mod FooBar { //~ ERROR module `FooBar` should have a snake case name such as `foo_bar`
    pub struct S;
}

fn f(_: FooBar::S) { }

fn main() { }
