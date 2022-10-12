// This test checks an optimization that is not guaranteed to work. This test case should not block
// a future LLVM update.
// compile-flags: -O

#![crate_type = "lib"]

pub enum Bar {
    A = 1,
    B = 3,
}

// CHECK-LABEL: @lookup_inc
#[no_mangle]
pub fn lookup_inc(buf: &[u8; 5], f: Bar) -> u8 {
    // FIXME: panic check can be removed by adding the assumes back after https://github.com/rust-lang/rust/pull/98332
    // CHECK: panic_bounds_check
    buf[f as usize + 1]
}

// CHECK-LABEL: @lookup_dec
#[no_mangle]
pub fn lookup_dec(buf: &[u8; 5], f: Bar) -> u8 {
    // FIXME: panic check can be removed by adding the assumes back after https://github.com/rust-lang/rust/pull/98332
    // CHECK: panic_bounds_check
    buf[f as usize - 1]
}
