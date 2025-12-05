//@ compile-flags: -Copt-level=3
//@ only-x86_64 (vectorization varies between architectures)
#![crate_type = "lib"]

// Ensure that slice + take + sum gets vectorized.
// Currently this relies on the slice::Iter::try_fold implementation
// CHECK-LABEL: @slice_take_sum
#[no_mangle]
pub fn slice_take_sum(s: &[u64], l: usize) -> u64 {
    // CHECK: vector.body:
    // CHECK: ret
    s.iter().take(l).sum()
}
