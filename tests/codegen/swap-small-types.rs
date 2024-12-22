//@ compile-flags: -O -Z merge-functions=disabled
//@ only-x86_64

#![crate_type = "lib"]

use std::mem::swap;

type RGB48 = [u16; 3];

// CHECK-LABEL: @swap_rgb48_manually(
#[no_mangle]
pub fn swap_rgb48_manually(x: &mut RGB48, y: &mut RGB48) {
    // FIXME: See #115212 for why this has an alloca again

    // CHECK: alloca [6 x i8], align 2
    // CHECK: call void @llvm.memcpy.p0.p0.i64({{.+}}, i64 6, i1 false)
    // CHECK: call void @llvm.memcpy.p0.p0.i64({{.+}}, i64 6, i1 false)
    // CHECK: call void @llvm.memcpy.p0.p0.i64({{.+}}, i64 6, i1 false)

    let temp = *x;
    *x = *y;
    *y = temp;
}

// CHECK-LABEL: @swap_rgb48
#[no_mangle]
pub fn swap_rgb48(x: &mut RGB48, y: &mut RGB48) {
    // CHECK-NOT: alloca

    // Swapping `i48` might be cleaner in LLVM-IR here, but `i32`+`i16` isn't bad,
    // and is closer to the assembly it generates anyway.

    // CHECK-NOT: load
    // CHECK: load i32{{.+}}align 2
    // CHECK-NEXT: load i32{{.+}}align 2
    // CHECK-NEXT: store i32{{.+}}align 2
    // CHECK-NEXT: store i32{{.+}}align 2
    // CHECK: load i16{{.+}}align 2
    // CHECK-NEXT: load i16{{.+}}align 2
    // CHECK-NEXT: store i16{{.+}}align 2
    // CHECK-NEXT: store i16{{.+}}align 2
    // CHECK-NOT: store
    swap(x, y)
}

type RGBA64 = [u16; 4];

// CHECK-LABEL: @swap_rgba64
#[no_mangle]
pub fn swap_rgba64(x: &mut RGBA64, y: &mut RGBA64) {
    // CHECK-NOT: alloca
    // CHECK-DAG: %[[XVAL:.+]] = load i64, ptr %x, align 2
    // CHECK-DAG: %[[YVAL:.+]] = load i64, ptr %y, align 2
    // CHECK-DAG: store i64 %[[YVAL]], ptr %x, align 2
    // CHECK-DAG: store i64 %[[XVAL]], ptr %y, align 2
    swap(x, y)
}

// CHECK-LABEL: @swap_vecs
#[no_mangle]
pub fn swap_vecs(x: &mut Vec<u32>, y: &mut Vec<u32>) {
    // CHECK-NOT: alloca

    // CHECK-NOT: load
    // CHECK: load i128
    // CHECK-NEXT: load i128
    // CHECK-NEXT: store i128
    // CHECK-NEXT: store i128
    // CHECK: load i64
    // CHECK-NEXT: load i64
    // CHECK-NEXT: store i64
    // CHECK-NEXT: store i64
    // CHECK-NOT: store
    swap(x, y)
}

// CHECK-LABEL: @swap_slices
#[no_mangle]
pub fn swap_slices<'a>(x: &mut &'a [u32], y: &mut &'a [u32]) {
    // Note that separate loads here is fine, as they merge to `movups` anyway
    // at the assembly level, so staying more obviously typed and as a scalar
    // pair -- like they're used elsewhere -- is ok, no need to force `i128`.

    // CHECK-NOT: alloca
    // CHECK: load ptr
    // CHECK: load i64
    // CHECK: call void @llvm.memcpy.p0.p0.i64({{.+}}, i64 16, i1 false)
    // CHECK: store ptr
    // CHECK: store i64
    swap(x, y)
}

type RGB24 = [u8; 3];

// CHECK-LABEL: @swap_rgb24_slices
#[no_mangle]
pub fn swap_rgb24_slices(x: &mut [RGB24], y: &mut [RGB24]) {
    // CHECK-NOT: alloca

    // The odd size means we need the full set.

    // CHECK-COUNT-2: load i512{{.+}}align 1
    // CHECK-NEXT: store i512{{.+}}align 1
    // CHECK-COUNT-2: load i256{{.+}}align 1
    // CHECK-NEXT: store i256{{.+}}align 1
    // CHECK-COUNT-2: load i128{{.+}}align 1
    // CHECK-NEXT: store i128{{.+}}align 1
    // CHECK-COUNT-2: load i64{{.+}}align 1
    // CHECK-NEXT: store i64{{.+}}align 1
    // CHECK-COUNT-2: load i32{{.+}}align 1
    // CHECK-NEXT: store i32{{.+}}align 1
    // CHECK-COUNT-2: load i16{{.+}}align 1
    // CHECK-NEXT: store i16{{.+}}align 1
    // CHECK-COUNT-2: load i8{{.+}}align 1
    // CHECK-NEXT: store i8{{.+}}align 1
    if x.len() == y.len() {
        x.swap_with_slice(y);
    }
}

type RGBA32 = [u8; 4];

// CHECK-LABEL: @swap_rgba32_slices
#[no_mangle]
pub fn swap_rgba32_slices(x: &mut [RGBA32], y: &mut [RGBA32]) {
    // CHECK-NOT: alloca

    // Because the size in bytes in a multiple of 4, we can skip the smallest sizes.

    // CHECK-COUNT-2: load i512{{.+}}align 1
    // CHECK-NEXT: store i512{{.+}}align 1
    // CHECK-COUNT-2: load i256{{.+}}align 1
    // CHECK-NEXT: store i256{{.+}}align 1
    // CHECK-COUNT-2: load i128{{.+}}align 1
    // CHECK-NEXT: store i128{{.+}}align 1
    // CHECK-COUNT-2: load i64{{.+}}align 1
    // CHECK-NEXT: store i64{{.+}}align 1
    // CHECK-COUNT-2: load i32{{.+}}align 1
    // CHECK-NEXT: store i32{{.+}}align 1
    // CHECK-NOT: load i16
    // CHECK-NOT: store i16
    // CHECK-NOT: load i8
    // CHECK-NOT: store i8
    if x.len() == y.len() {
        x.swap_with_slice(y);
    }
}

// Strings have a non-power-of-two size, but have pointer alignment.
const _: () = assert!(!std::mem::size_of::<String>().is_power_of_two());

// CHECK-LABEL: @swap_string_slices
#[no_mangle]
pub fn swap_string_slices(x: &mut [String], y: &mut [String]) {
    // CHECK-NOT: alloca

    // CHECK-COUNT-2: load i512{{.+}}align 8
    // CHECK-NEXT: store i512{{.+}}align 8
    // CHECK-COUNT-2: load i256{{.+}}align 8
    // CHECK-NEXT: store i256{{.+}}align 8
    // CHECK-COUNT-2: load i128{{.+}}align 8
    // CHECK-NEXT: store i128{{.+}}align 8
    // CHECK-COUNT-2: load i64{{.+}}align 8
    // CHECK-NEXT: store i64{{.+}}align 8
    // CHECK-NOT: load i32
    // CHECK-NOT: store i32
    // CHECK-NOT: load i16
    // CHECK-NOT: store i16
    // CHECK-NOT: load i8
    // CHECK-NOT: store i8
    if x.len() == y.len() {
        x.swap_with_slice(y);
    }
}

#[repr(C, packed)]
pub struct Packed {
    pub first: bool,
    pub second: usize,
}

// CHECK-LABEL: @swap_packed_structs
#[no_mangle]
pub fn swap_packed_structs(x: &mut Packed, y: &mut Packed) {
    // CHECK-NOT: alloca
    // CHECK-COUNT-2: load i64{{.+}}align 1
    // CHECK-COUNT-2: store i64{{.+}}align 1
    // CHECK-COUNT-2: load i8{{.+}}align 1
    // CHECK-COUNT-2: store i8{{.+}}align 1
    swap(x, y)
}
