#![feature(const_generics)]
//~^ ERROR features `const_generics` and `min_const_generics` are incompatible
#![allow(incomplete_features)]
#![feature(min_const_generics)]


fn main() {}
