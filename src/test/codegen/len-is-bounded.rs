// min-llvm-version: 11.0
// compile-flags: -O -C panic=abort
#![crate_type = "lib"]

#[no_mangle]
pub fn len_range(a: &[u8], b: &[u8]) -> usize {
    // CHECK-NOT: panic
    a.len().checked_add(b.len()).unwrap()
}

#[no_mangle]
pub fn len_range_on_non_byte(a: &[u16], b: &[u16]) -> usize {
    // CHECK-NOT: panic
    a.len().checked_add(b.len()).unwrap()
}

pub struct Zst;

#[no_mangle]
pub fn zst_range(a: &[Zst], b: &[Zst]) -> usize {
    // Zsts may be arbitrarily large.
    // CHECK: panic
    a.len().checked_add(b.len()).unwrap()
}
