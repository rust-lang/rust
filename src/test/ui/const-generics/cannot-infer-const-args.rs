#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

fn foo<const X: usize>() -> usize {
    0
}

fn main() {
    foo(); //~ ERROR type annotations needed
}
