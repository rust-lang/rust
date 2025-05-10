// Regression test for #82792.

//@ run-pass

#[repr(C)]
#[allow(improper_ctype_definitions)]
pub struct Loaf<T: Sized, const N: usize = 1> {
    head: [T; N],
    slice: [T],
}

fn main() {}
