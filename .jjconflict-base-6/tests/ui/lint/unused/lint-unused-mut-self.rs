//@ run-rustfix

#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![deny(unused_mut)]

struct Foo;
impl Foo {
    fn foo(mut self) {} //~ ERROR: variable does not need to be mutable
    fn bar(mut self: Box<Foo>) {} //~ ERROR: variable does not need to be mutable
}

fn main() {}
