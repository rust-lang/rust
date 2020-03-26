#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

fn foo<const N: usize>() {
    let _ = [0u64; N + 1];
    //~^ ERROR constant expression depends on a generic parameter
}

fn main() {}
