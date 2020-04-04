#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

fn is_123<const N: usize>(x: [u32; N]) -> bool {
    match x {
        [1, 2] => true, //~ ERROR mismatched types
        _ => false
    }
}

fn main() {}
