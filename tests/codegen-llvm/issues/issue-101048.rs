//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

#[no_mangle]
pub fn all_zero(data: &[u64]) -> bool {
    // CHECK-LABEL: @all_zero(
    // CHECK: [[PHI:%.*]] = phi i1
    // CHECK-NOT: phi i8
    // CHECK-NOT: zext
    data.iter().copied().fold(true, |acc, x| acc & (x == 0))
}
