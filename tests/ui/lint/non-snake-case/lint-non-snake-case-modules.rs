#![deny(non_snake_case)]
#![allow(dead_code)]

mod FooBar1 {
    //~^ ERROR module `FooBar1` should have a snake case name
    pub struct S;
}

fn f(_: FooBar1::S) {}

mod foo_bar2 {
    pub struct S;
}
use foo_bar2 as FooBar2; //~ ERROR renamed module `FooBar2` should have a snake case name

fn main() {}
