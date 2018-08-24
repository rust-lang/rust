#![allow(warnings)]

#![feature(impl_header_lifetime_elision)]

trait MyTrait { }

struct Foo<'a> { x: &'a u32 }

impl MyTrait for Foo {
    //~^ ERROR missing lifetime specifier
}

fn main() {}
