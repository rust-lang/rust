// compile-flags: -Zmir-opt-level=4
// run-pass

#![feature(const_generics)]
#![allow(incomplete_features)]
fn main() {
    fn foo<const N: usize>() -> [u8; N] {
        [0; N]
    }
    let _x = foo::<1>();
}
