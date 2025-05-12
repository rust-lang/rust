//@ compile-flags: -Copt-level=3 -Z merge-functions=disabled
//@ only-x86_64
//@ min-llvm-version: 20
//@ ignore-std-debug-assertions (`ptr::swap_nonoverlapping` has one which blocks some optimizations)

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

    // CHECK-NOT: load{{ }}
    // CHECK: load i32{{.+}}align 2
    // CHECK-NEXT: load i32{{.+}}align 2
    // CHECK-NEXT: store i32{{.+}}align 2
    // CHECK-NEXT: store i32{{.+}}align 2
    // CHECK: load i16{{.+}}align 2
    // CHECK-NEXT: load i16{{.+}}align 2
    // CHECK-NEXT: store i16{{.+}}align 2
    // CHECK-NEXT: store i16{{.+}}align 2
    // CHECK-NOT: store{{ }}
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
    // There are plenty more loads and stores than just these,
    // but at least one sure better be 64-bit (for size or capacity).
    // CHECK: load i64
    // CHECK: load i64
    // CHECK: store i64
    // CHECK: store i64
    // CHECK: ret void
    swap(x, y)
}

// CHECK-LABEL: @swap_slices
#[no_mangle]
pub fn swap_slices<'a>(x: &mut &'a [u32], y: &mut &'a [u32]) {
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

    // CHECK: mul nuw nsw i64 %{{x|y}}.1, 3

    // CHECK: load <{{[0-9]+}} x i64>
    // CHECK: store <{{[0-9]+}} x i64>

    // CHECK-COUNT-2: load i32
    // CHECK-COUNT-2: store i32
    // CHECK-COUNT-2: load i16
    // CHECK-COUNT-2: store i16
    // CHECK-COUNT-2: load i8
    // CHECK-COUNT-2: store i8
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

    // CHECK: load <{{[0-9]+}} x i64>
    // CHECK: store <{{[0-9]+}} x i64>

    // CHECK-COUNT-2: load i32
    // CHECK-COUNT-2: store i32

    // CHECK-NOT: load i16
    // CHECK-NOT: store i16
    // CHECK-NOT: load i8
    // CHECK-NOT: store i8

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
    // CHECK: load <{{[0-9]+}} x i64>{{.+}}, align 8,
    // CHECK: store <{{[0-9]+}} x i64>{{.+}}, align 8,
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

    // CHECK-NOT: load
    // CHECK-NOT: store

    // CHECK: %[[A:.+]] = load i64, ptr %x, align 1,
    // CHECK-NEXT: %[[B:.+]] = load i64, ptr %y, align 1,
    // CHECK-NEXT: store i64 %[[B]], ptr %x, align 1,
    // CHECK-NEXT: store i64 %[[A]], ptr %y, align 1,

    // CHECK-NOT: load
    // CHECK-NOT: store

    // CHECK: %[[C:.+]] = load i8, ptr %[[X8:.+]], align 1,
    // CHECK-NEXT: %[[D:.+]] = load i8, ptr %[[Y8:.+]], align 1,
    // CHECK-NEXT: store i8 %[[D]], ptr %[[X8]], align 1,
    // CHECK-NEXT: store i8 %[[C]], ptr %[[Y8]], align 1,

    // CHECK-NOT: load
    // CHECK-NOT: store

    // CHECK: ret void
    swap(x, y)
}
