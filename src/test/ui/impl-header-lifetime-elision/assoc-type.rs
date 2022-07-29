// Test that we do not yet support elision in associated types, even
// when there is just one name we could take from the impl header.

#![allow(warnings)]

trait MyTrait {
    type Output;
}

impl MyTrait for &i32 {
    type Output = &i32;
    //~^ ERROR missing lifetime specifier
}

impl MyTrait for &u32 {
    type Output = &'_ i32;
    //~^ ERROR missing lifetime specifier
}

// This is what you have to do:
impl<'a> MyTrait for &'a f32 {
    type Output = &'a f32;
}

fn main() { }
