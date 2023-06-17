// compile-flags: -O -Z merge-functions=disabled
// only-x86_64
// ignore-debug: the debug assertions get in the way

#![crate_type = "lib"]

use std::mem::swap;

// CHECK-LABEL: @swap_fat_ptrs
#[no_mangle]
pub fn swap_fat_ptrs<'a>(x: &mut &'a str, y: &mut &'a str) {
    // CHECK-NOT: alloca
    // CHECK: %[[X0:.+]] = load ptr, ptr %x, align 8
    // CHECK: %[[X1:.+]] = load i64, ptr %[[PX1:.+]], align 8
    // CHECK: %[[Y0:.+]] = load ptr, ptr %y, align 8
    // CHECK: %[[Y1:.+]] = load i64, ptr %[[PY1:.+]], align 8
    // CHECK: store ptr %[[Y0]], ptr %x, align 8
    // CHECK: store i64 %[[Y1]], ptr %[[PX1]], align 8
    // CHECK: store ptr %[[X0]], ptr %y, align 8
    // CHECK: store i64 %[[X1]], ptr %[[PY1]], align 8
    swap(x, y)
}

type RGB48 = [u16; 3];

// CHECK-LABEL: @swap_rgb48_manually(
#[no_mangle]
pub fn swap_rgb48_manually(x: &mut RGB48, y: &mut RGB48) {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP0:.+]] = load <3 x i16>, ptr %x, align 2
    // CHECK: %[[TEMP1:.+]] = load <3 x i16>, ptr %y, align 2
    // CHECK: store <3 x i16> %[[TEMP1]], ptr %x, align 2
    // CHECK: store <3 x i16> %[[TEMP0]], ptr %y, align 2

    let temp = *x;
    *x = *y;
    *y = temp;
}

// CHECK-LABEL: @swap_rgb48
#[no_mangle]
pub fn swap_rgb48(x: &mut RGB48, y: &mut RGB48) {
    // CHECK-NOT: alloca
    // CHECK: load <3 x i16>
    // CHECK: load <3 x i16>
    // CHECK: store <3 x i16>
    // CHECK: store <3 x i16>
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

// CHECK-LABEL: @swap_rgba32
#[no_mangle]
pub fn swap_rgba32(x: &mut RGBA32, y: &mut RGBA32) {
    // CHECK-NOT: alloca
    // CHECK: load <4 x i8>
    // CHECK: load <4 x i8>
    // CHECK: store <4 x i8>
    // CHECK: store <4 x i8>
    swap(x, y)
}

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

// It's wasteful to do three `memcpy`s when a `String` is just three fields.

// CHECK-LABEL: @swap_strings
#[no_mangle]
pub fn swap_strings(x: &mut String, y: &mut String) {
    // CHECK-NOT: alloca
    // CHECK: load i192
    // CHECK: load i192
    // CHECK: store i192
    // CHECK: store i192
    swap(x, y)
}

// CHECK-LABEL: @swap_tuple_with_padding
#[no_mangle]
pub fn swap_tuple_with_padding(x: &mut (u8, u32, u8), y: &mut (u8, u32, u8)) {
    // CHECK-NOT: alloca
    // CHECK: load i64
    // CHECK-NOT: noundef
    // CHECK: load i64
    // CHECK-NOT: noundef
    // CHECK: store i64
    // CHECK: store i64
    swap(x, y)
}
