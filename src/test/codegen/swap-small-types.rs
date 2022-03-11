// compile-flags: -O
// only-x86_64
// ignore-debug: the debug assertions get in the way

#![crate_type = "lib"]

use std::mem::swap;

type RGB48 = [u16; 3];

// CHECK-LABEL: @swap_rgb48
#[no_mangle]
pub fn swap_rgb48(x: &mut RGB48, y: &mut RGB48) {
// CHECK-NOT: alloca
// CHECK: load i48
// CHECK: store i48
    swap(x, y)
}

// LLVM doesn't vectorize a loop over 3-byte elements,
// so we chunk it down to bytes and loop over those instead.
type RGB24 = [u8; 3];

// CHECK-LABEL: @swap_rgb24_slices
#[no_mangle]
pub fn swap_rgb24_slices(x: &mut [RGB24], y: &mut [RGB24]) {
// CHECK-NOT: alloca
// CHECK: load <{{[0-9]+}} x i8>
// CHECK: store <{{[0-9]+}} x i8>
    if x.len() == y.len() {
        x.swap_with_slice(y);
    }
}

// This one has a power-of-two size, so we iterate over it directly
type RGBA32 = [u8; 4];

// CHECK-LABEL: @swap_rgba32_slices
#[no_mangle]
pub fn swap_rgba32_slices(x: &mut [RGBA32], y: &mut [RGBA32]) {
// CHECK-NOT: alloca
// CHECK: load <{{[0-9]+}} x i32>
// CHECK: store <{{[0-9]+}} x i32>
    if x.len() == y.len() {
        x.swap_with_slice(y);
    }
}

// Strings have a non-power-of-two size, but have pointer alignment,
// so we swap usizes instead of dropping all the way down to bytes.
const _: () = assert!(!std::mem::size_of::<String>().is_power_of_two());

// CHECK-LABEL: @swap_string_slices
#[no_mangle]
pub fn swap_string_slices(x: &mut [String], y: &mut [String]) {
// CHECK-NOT: alloca
// CHECK: load <{{[0-9]+}} x i64>
// CHECK: store <{{[0-9]+}} x i64>
    if x.len() == y.len() {
        x.swap_with_slice(y);
    }
}
