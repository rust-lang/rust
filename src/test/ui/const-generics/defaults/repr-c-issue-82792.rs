// Regression test for #82792.

// run-pass

#![feature(const_generics_defaults)]

#[repr(C)]
pub struct Loaf<T: Sized, const N: usize = 1> {
    head: [T; N],
    slice: [T],
}

fn main() {}
