#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

struct S<const C: u8>(C); //~ ERROR expected type, found const parameter

fn main() {}
