#![allow(warnings)]

// Make sure this related feature didn't accidentally enable this
#![feature(in_band_lifetimes)]

trait MyTrait<'a> { }

impl MyTrait<'a> for &u32 { }
//~^ ERROR missing lifetime specifier

struct MyStruct;
trait MarkerTrait {}

impl MarkerTrait for &'_ MyStruct { }
//~^ ERROR missing lifetime specifier

fn main() {}
