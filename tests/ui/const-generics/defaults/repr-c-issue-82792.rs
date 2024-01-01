// Regression test for #82792.

// build-pass

#[repr(C)]
pub struct Loaf<T: Sized, const N: usize = 1> {
    head: [T; N],
    slice: [T],
}

fn main() {}
