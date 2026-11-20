#![allow(warnings)]

trait MyTrait<'a> {}

impl MyTrait for u32 {}
//~^ ERROR implicit elided lifetime not allowed here

fn main() {}
