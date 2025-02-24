//@ compile-flags: -Copt-level=3
//@ only-x86_64

#![crate_type = "lib"]

use std::mem::swap;
use std::ptr::{copy_nonoverlapping, read, write};

type KeccakBuffer = [[u64; 5]; 5];

// A basic read+copy+write swap implementation ends up copying one of the values
// to stack for large types, which is completely unnecessary as the lack of
// overlap means we can just do whatever fits in registers at a time.

// CHECK-LABEL: @swap_basic
#[no_mangle]
pub fn swap_basic(x: &mut KeccakBuffer, y: &mut KeccakBuffer) {
    // CHECK: alloca [200 x i8]

    // SAFETY: exclusive references are always valid to read/write,
    // are non-overlapping, and nothing here panics so it's drop-safe.
    unsafe {
        let z = read(x);
        copy_nonoverlapping(y, x, 1);
        write(y, z);
    }
}

// This test verifies that the library does something smarter, and thus
// doesn't need any scratch space on the stack.

// CHECK-LABEL: @swap_std
#[no_mangle]
pub fn swap_std(x: &mut KeccakBuffer, y: &mut KeccakBuffer) {
    // CHECK-NOT: alloca
    // CHECK: load <{{[0-9]+}} x i64>
    // CHECK: store <{{[0-9]+}} x i64>
    swap(x, y)
}

// CHECK-LABEL: @swap_slice
#[no_mangle]
pub fn swap_slice(x: &mut [KeccakBuffer], y: &mut [KeccakBuffer]) {
    // CHECK-NOT: alloca
    // CHECK: load <{{[0-9]+}} x i64>
    // CHECK: store <{{[0-9]+}} x i64>
    if x.len() == y.len() {
        x.swap_with_slice(y);
    }
}

type OneKilobyteBuffer = [u8; 1024];

// CHECK-LABEL: @swap_1kb_slices
#[no_mangle]
pub fn swap_1kb_slices(x: &mut [OneKilobyteBuffer], y: &mut [OneKilobyteBuffer]) {
    // CHECK-NOT: alloca

    // CHECK-NOT: load i32
    // CHECK-NOT: store i32
    // CHECK-NOT: load i16
    // CHECK-NOT: store i16
    // CHECK-NOT: load i8
    // CHECK-NOT: store i8

    // CHECK: load <{{[0-9]+}} x i64>{{.+}}align 1,
    // CHECK: store <{{[0-9]+}} x i64>{{.+}}align 1,

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

#[repr(align(64))]
pub struct BigButHighlyAligned([u8; 64 * 3]);

// CHECK-LABEL: @swap_big_aligned
#[no_mangle]
pub fn swap_big_aligned(x: &mut BigButHighlyAligned, y: &mut BigButHighlyAligned) {
    // CHECK-NOT: call void @llvm.memcpy
    // CHECK-NOT: load i32
    // CHECK-NOT: store i32
    // CHECK-NOT: load i16
    // CHECK-NOT: store i16
    // CHECK-NOT: load i8
    // CHECK-NOT: store i8

    // CHECK-COUNT-2: load <{{[0-9]+}} x i64>{{.+}}align 64,
    // CHECK-COUNT-2: store <{{[0-9]+}} x i64>{{.+}}align 64,

    // CHECK-COUNT-2: load <{{[0-9]+}} x i64>{{.+}}align 32,
    // CHECK-COUNT-2: store <{{[0-9]+}} x i64>{{.+}}align 32,

    // CHECK-NOT: load i32
    // CHECK-NOT: store i32
    // CHECK-NOT: load i16
    // CHECK-NOT: store i16
    // CHECK-NOT: load i8
    // CHECK-NOT: store i8
    // CHECK-NOT: call void @llvm.memcpy
    swap(x, y)
}
