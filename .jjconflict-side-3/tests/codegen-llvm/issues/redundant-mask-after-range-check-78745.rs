// Tests that a bit mask is elided when a preceding range check or clamp already
// guarantees the masked bits are clear.
// See <https://github.com/rust-lang/rust/issues/78745>.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @redundant_mask_after_range_check(
// CHECK-NOT: and i32
// CHECK: ret i32
#[no_mangle]
pub fn redundant_mask_after_range_check(mut u: u32) -> u32 {
    if u <= 0x3F {
        u &= 0x7F;
    }
    u
}

// CHECK-LABEL: @redundant_mask_after_clamp(
// CHECK-NOT: and i32
// CHECK: ret i32
#[no_mangle]
pub fn redundant_mask_after_clamp(mut u: u32) -> u32 {
    if u > 0x7F {
        u = 0x7F;
    }
    u &= 0x7F;
    u
}
