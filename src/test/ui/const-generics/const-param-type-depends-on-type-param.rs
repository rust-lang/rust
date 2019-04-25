#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

pub struct Dependent<T, const X: T>([(); X]); //~ ERROR const parameters
//~^ ERROR parameter `T` is never used

fn main() {}
