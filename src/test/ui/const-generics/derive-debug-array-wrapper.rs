#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

#[derive(Debug)]
struct X<const N: usize> {
    a: [u32; N], //~ ERROR arrays only have std trait implementations for lengths 0..=32
}

fn main() {}
