#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

struct A<const N: &u8>;
//~^ ERROR `&` without an explicit lifetime name cannot be used here
trait B {}

impl<const N: &u8> A<N> { //~ ERROR `&` without an explicit lifetime name cannot be used here
    fn foo<const M: &u8>(&self) {}
    //~^ ERROR `&` without an explicit lifetime name cannot be used here
}

impl<const N: &u8> B for A<N> {}
//~^ ERROR `&` without an explicit lifetime name cannot be used here

fn bar<const N: &u8>() {}
//~^ ERROR `&` without an explicit lifetime name cannot be used here

fn main() {}
