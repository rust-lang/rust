#![allow(warnings)]

trait MyTrait<'a> { }

impl MyTrait for u32 {
    //~^ ERROR missing lifetime specifier
}

fn main() {}
