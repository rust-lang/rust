#![allow(warnings)]

trait MyTrait<'a> { }

impl<'a> MyTrait<'a> for &u32 { }
//~^ ERROR missing lifetime specifier

impl<'a> MyTrait<'_> for &'a f32 { }
//~^ ERROR missing lifetime specifier

fn main() {}
