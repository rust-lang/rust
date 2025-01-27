//@ compile-flags: -O
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
    // CHECK-COUNT-2: load i512{{.+}}align 8
    // CHECK-COUNT-2: store i512{{.+}}align 8
    // CHECK-COUNT-2: load i512{{.+}}align 8
    // CHECK-COUNT-2: store i512{{.+}}align 8
    // CHECK-COUNT-2: load i512{{.+}}align 8
    // CHECK-COUNT-2: store i512{{.+}}align 8
    // CHECK-COUNT-2: load i64{{.+}}align 8
    // CHECK-COUNT-2: store i64{{.+}}align 8
    swap(x, y)
}

type OneKilobyteBuffer = [u8; 1024];

// CHECK-LABEL: @swap_1kb_slices
#[no_mangle]
pub fn swap_1kb_slices(x: &mut [OneKilobyteBuffer], y: &mut [OneKilobyteBuffer]) {
    // CHECK-NOT: alloca

    // These are so big that there's only the biggest chunk size used

    // CHECK-NOT: load i256
    // CHECK-NOT: load i128
    // CHECK-NOT: load i64
    // CHECK-NOT: load i32
    // CHECK-NOT: load i16
    // CHECK-NOT: load i8

    // CHECK-COUNT-2: load i512{{.+}}align 1
    // CHECK-COUNT-2: store i512{{.+}}align 1

    // CHECK-NOT: store i256
    // CHECK-NOT: store i128
    // CHECK-NOT: store i64
    // CHECK-NOT: store i32
    // CHECK-NOT: store i16
    // CHECK-NOT: store i8
    if x.len() == y.len() {
        x.swap_with_slice(y);
    }
}

// This verifies that the 2×read + 2×write optimizes to just 3 memcpys
// for an unusual type like this.  It's not clear whether we should do anything
// smarter in Rust for these, so for now it's fine to leave these up to the backend.
// That's not as bad as it might seem, as for example, LLVM will lower the
// memcpys below to VMOVAPS on YMMs if one enables the AVX target feature.
// Eventually we'll be able to pass `align_of::<T>` to a const generic and
// thus pick a smarter chunk size ourselves without huge code duplication.

#[repr(align(64))]
pub struct BigButHighlyAligned([u8; 64 * 3]);

// CHECK-LABEL: @swap_big_aligned
#[no_mangle]
pub fn swap_big_aligned(x: &mut BigButHighlyAligned, y: &mut BigButHighlyAligned) {
    // CHECK-NOT: alloca
    // CHECK-COUNT-2: load i512{{.+}}align 64
    // CHECK-COUNT-2: store i512{{.+}}align 64
    // CHECK-COUNT-2: load i512{{.+}}align 64
    // CHECK-COUNT-2: store i512{{.+}}align 64
    // CHECK-COUNT-2: load i512{{.+}}align 64
    // CHECK-COUNT-2: store i512{{.+}}align 64
    swap(x, y)
}
