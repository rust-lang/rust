#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

fn foo<const N: usize>() {
    let _ = [0u64; N + 1];
    //~^ ERROR array lengths can't depend on generic parameters
}

fn main() {}
