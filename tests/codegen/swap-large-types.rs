// compile-flags: -O
// only-x86_64
// ignore-debug: the debug assertions get in the way

#![crate_type = "lib"]

use std::mem::swap;
use std::ptr::{read, copy_nonoverlapping, write};

type KeccakBuffer = [[u64; 5]; 5];

// A basic read+copy+write swap implementation ends up copying one of the values
// to stack for large types, which is completely unnecessary as the lack of
// overlap means we can just do whatever fits in registers at a time.

// CHECK-LABEL: @swap_basic
#[no_mangle]
pub fn swap_basic(x: &mut KeccakBuffer, y: &mut KeccakBuffer) {
// CHECK: alloca [5 x [5 x i64]]

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

// Verify that types with usize alignment are swapped via vectored usizes,
// not falling back to byte-level code.

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

// But for a large align-1 type, vectorized byte copying is what we want.

type OneKilobyteBuffer = [u8; 1024];

// CHECK-LABEL: @swap_1kb_slices
#[no_mangle]
pub fn swap_1kb_slices(x: &mut [OneKilobyteBuffer], y: &mut [OneKilobyteBuffer]) {
// CHECK-NOT: alloca
// CHECK: load <{{[0-9]+}} x i8>
// CHECK: store <{{[0-9]+}} x i8>
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
// CHECK-NOT: call void @llvm.memcpy
// CHECK: call void @llvm.memcpy.{{.+}}(ptr noundef nonnull align 64 dereferenceable(192)
// CHECK: call void @llvm.memcpy.{{.+}}(ptr noundef nonnull align 64 dereferenceable(192)
// CHECK: call void @llvm.memcpy.{{.+}}(ptr noundef nonnull align 64 dereferenceable(192)
// CHECK-NOT: call void @llvm.memcpy
    swap(x, y)
}
