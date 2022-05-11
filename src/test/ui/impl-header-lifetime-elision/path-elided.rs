#![deny(elided_lifetimes_in_paths)]

trait MyTrait { }

struct Foo<'a> { x: &'a u32 }

impl MyTrait for Foo {
    //~^ ERROR hidden lifetime parameters in types are deprecated [elided_lifetimes_in_paths]
}

fn main() {}
