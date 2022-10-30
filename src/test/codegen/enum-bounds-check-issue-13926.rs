// This test checks an optimization that is not guaranteed to work. This test case should not block
// a future LLVM update.
// compile-flags: -O

#![crate_type = "lib"]

#[repr(u8)]
pub enum Exception {
    Low = 5,
    High = 10,
}

// CHECK-LABEL: @access
#[no_mangle]
pub fn access(array: &[usize; 12], exc: Exception) -> usize {
    // FIXME: panic check can be removed by adding the assumes back after https://github.com/rust-lang/rust/pull/98332
    // CHECK: panic_bounds_check
    array[(exc as u8 - 4) as usize]
}
