#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

fn bar<const X: (), 'a>(_: &'a ()) {} //~ ERROR incorrect parameter order
fn foo<const X: (), T>(_: &T) {} //~ ERROR incorrect parameter order

fn main() {}
