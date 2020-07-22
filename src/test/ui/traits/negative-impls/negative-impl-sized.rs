#![feature(negative_impls)]

// Test a negative impl for a trait requires `T: ?Sized`.

trait MyTrait {}

impl<T> !MyTrait for T {} //~ ERROR auto traits must not contain where bounds

fn main() {}
