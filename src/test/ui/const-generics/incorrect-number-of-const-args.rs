#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

fn foo<const X: usize, const Y: usize>() -> usize {
    0
}

fn main() {
    foo::<0>(); //~ ERROR wrong number of const arguments: expected 2, found 1
    foo::<0, 0, 0>(); //~ ERROR wrong number of const arguments: expected 2, found 3
}
