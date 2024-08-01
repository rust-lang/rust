// Regression test for #82792.

//@ run-pass

#[allow(dead_code)]
#[repr(C)]
pub struct Loaf<T: Sized, const N: usize = 1> {
    head: [T; N],
    slice: [T],
}

fn main() {}
