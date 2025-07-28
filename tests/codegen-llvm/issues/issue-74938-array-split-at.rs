//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

const N: usize = 3;
pub type T = u8;

// CHECK-LABEL: @split_multiple
// CHECK-NOT: unreachable
#[no_mangle]
pub fn split_multiple(slice: &[T]) -> (&[T], &[T]) {
    let len = slice.len() / N;
    slice.split_at(len * N)
}
