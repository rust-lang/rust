#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

#[derive(Debug)]
struct X<const N: usize> {
    a: [u32; N], //~ ERROR `[u32; _]` doesn't implement `std::fmt::Debug`
}

fn main() {}
