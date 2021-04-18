// Regression test for #82792.

// run-pass

#![feature(const_generics_defaults)]
#![allow(incomplete_features)]

#[repr(C)]
pub struct Loaf<T: Sized, const N: usize = 1> {
    head: [T; N],
    slice: [T],
}

fn main() {}
