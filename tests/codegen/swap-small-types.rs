// compile-flags: -Copt-level=3 -Z merge-functions=disabled
// only-x86_64
// ignore-debug: the debug assertions get in the way

#![crate_type = "lib"]
#![feature(portable_simd)]

use std::mem::swap;

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
    // CHECK-NOT: br
    // CHECK: load i32

    // CHECK-NOT: alloca
    // CHECK-NOT: br
    // CHECK: store i32

    // CHECK-NOT: alloca
    // CHECK-NOT: br
    // CHECK: load i16

    // CHECK-NOT: alloca
    // CHECK-NOT: br
    // CHECK: store i16

    // CHECK-NOT: alloca
    // CHECK-NOT: br
    // CHECK: ret void
    swap(x, y)
}

// CHECK-LABEL: @swap_vecs
#[no_mangle]
pub fn swap_vecs(x: &mut Vec<u32>, y: &mut Vec<u32>) {
    // CHECK-NOT: alloca
    // CHECK-NOT: br

    // CHECK: load <{{[0-9]+}} x i64>
    // CHECK-NOT: alloca
    // CHECK-NOT: br

    // CHECK: store <{{[0-9]+}} x i64>
    // CHECK-NOT: alloca
    // CHECK-NOT: br

    // CHECK: load i64
    // CHECK-NOT: alloca
    // CHECK-NOT: br

    // CHECK: store i64
    // CHECK-NOT: alloca
    // CHECK-NOT: br

    // CHECK: ret void
    swap(x, y)
}

// CHECK-LABEL: @swap_slices
#[no_mangle]
pub fn swap_slices<'a>(x: &mut &'a [u32], y: &mut &'a [u32]) {
    // CHECK-NOT: alloca
    // CHECK-NOT: br
    // CHECK: load <{{[0-9]+}} x i64>
    // CHECK: store <{{[0-9]+}} x i64>
    // CHECK: ret void
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

// This one has a power-of-two size, so we iterate over it using ints.
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

#[repr(C, packed)]
pub struct Packed {
    pub first: bool,
    pub second: u64,
}

// CHECK-LABEL: @swap_packed_structs
#[no_mangle]
pub fn swap_packed_structs(x: &mut Packed, y: &mut Packed) {
    // CHECK-NOT: alloca
    // CHECK: ret void
    swap(x, y)
}

// CHECK-LABEL: @swap_simd_type
#[no_mangle]
pub fn swap_simd_type(x: &mut std::simd::f32x4, y: &mut std::simd::f32x4){
    // CHECK-NOT: alloca
    // CHECK-NOT: br
    // CHECK: load <4 x float>

    // CHECK-NOT: alloca
    // CHECK-NOT: br
    // CHECK: store <4 x float>
    // CHECK: ret void
    swap(x, y)
}
