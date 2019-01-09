#![allow(warnings)]

trait MyTrait { }

struct Foo<'a> { x: &'a u32 }

impl MyTrait for Foo {
    //~^ ERROR missing lifetime specifier
}

fn main() {}
