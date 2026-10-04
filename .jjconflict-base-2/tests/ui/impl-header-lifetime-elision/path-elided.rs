#![allow(warnings)]

trait MyTrait { }

struct Foo<'a> { x: &'a u32 }

impl MyTrait for Foo {
    //~^ ERROR implicit elided lifetime not allowed here
}

fn main() {}
