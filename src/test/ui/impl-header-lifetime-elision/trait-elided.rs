#![deny(elided_lifetimes_in_paths)]

trait MyTrait<'a> {}

impl MyTrait for u32 {}
//~^ ERROR hidden lifetime parameters in types are deprecated [elided_lifetimes_in_paths]

fn main() {}
