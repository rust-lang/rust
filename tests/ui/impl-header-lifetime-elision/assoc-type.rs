// Test that we do not yet support elision in associated types, even
// when there is just one name we could take from the impl header.

#![allow(warnings)]

trait MyTrait {
    type Output;
}

impl MyTrait for &i32 {
    type Output = &i32;
    //~^ ERROR missing lifetime in associated type
}

impl MyTrait for &u32 {
    type Output = &'_ i32;
    //~^ ERROR `'_` cannot be used here
}

impl<'a> MyTrait for &f64 {
    type Output = &f64;
    //~^ ERROR missing lifetime in associated type
}

trait OtherTrait<'a> {
    type Output;
}
impl OtherTrait<'_> for f64 {
    type Output = &f64;
    //~^ ERROR missing lifetime in associated type
}

// This is what you have to do:
impl<'a> MyTrait for &'a f32 {
    type Output = &'a f32;
}

fn main() { }
