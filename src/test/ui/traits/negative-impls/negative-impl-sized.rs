#![feature(negative_impls)]

// Test that negative impls for a trait requires `T: ?Sized`.

trait MyTrait {}

impl<T> !MyTrait for T {}
//~^ ERROR negative impls on type parameters must not contain where bounds

fn main() {}
