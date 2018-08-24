#![allow(warnings)]

#![feature(impl_header_lifetime_elision)]

trait MyTrait<'a> { }

impl MyTrait for u32 {
    //~^ ERROR missing lifetime specifier
}

fn main() {}
