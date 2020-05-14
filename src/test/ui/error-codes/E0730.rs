#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

fn is_123<const N: usize>(x: [u32; N]) -> bool {
    match x {
        [1, 2, ..] => true, //~ ERROR cannot pattern-match on an array without a fixed length
        _ => false
    }
}

fn main() {}
