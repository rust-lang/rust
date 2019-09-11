#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

extern "C" {
    fn foo<const X: usize>(); //~ ERROR foreign items may not have const parameters

    fn bar<T, const X: usize>(_: T); //~ ERROR foreign items may not have type or const parameters
}

fn main() {}
