#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

// build-pass

fn foo<const N: usize>() {
    let _ = [0u64; N + 1];
}

fn main() {}
