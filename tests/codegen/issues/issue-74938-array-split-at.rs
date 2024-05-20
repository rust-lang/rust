//@ compile-flags: -O

#![crate_type = "lib"]

const N: usize = 3;
pub type T = u8;

#[no_mangle]
// CHECK-LABEL: @split_mutiple
// CHECK-NOT: unreachable
pub fn split_mutiple(slice: &[T]) -> (&[T], &[T]) {
    let len = slice.len() / N;
    slice.split_at(len * N)
}

