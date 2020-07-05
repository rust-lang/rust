// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

#[derive(Debug)]
struct X<const N: usize> {
    a: [u32; N],
}

fn main() {}
