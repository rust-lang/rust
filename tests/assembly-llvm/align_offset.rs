//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=1
//@ only-x86_64
#![crate_type = "rlib"]

// CHECK-LABEL: align_offset_byte_ptr
// CHECK: leaq 31
// CHECK: andq $-32
// CHECK: subq
#[no_mangle]
pub fn align_offset_byte_ptr(ptr: *const u8) -> usize {
    ptr.align_offset(32)
}

// CHECK-LABEL: align_offset_byte_slice
// CHECK: leaq 31
// CHECK: andq $-32
// CHECK: subq
#[no_mangle]
pub fn align_offset_byte_slice(slice: &[u8]) -> usize {
    slice.as_ptr().align_offset(32)
}

// CHECK-LABEL: align_offset_word_ptr
// CHECK: leaq 31
// CHECK: andq $-32
// CHECK: subq
// CHECK: shrq
// This `ptr` is not known to be aligned, so it is required to check if it is at all possible to
// align. LLVM applies a simple mask.
// CHECK: orq
#[no_mangle]
pub fn align_offset_word_ptr(ptr: *const u32) -> usize {
    ptr.align_offset(32)
}

// CHECK-LABEL: align_offset_word_slice
// CHECK: leaq 31
// CHECK: andq $-32
// CHECK: subq
// CHECK: shrq
// `slice` is known to be aligned, so `!0` is not possible as a return
// CHECK-NOT: orq
#[no_mangle]
pub fn align_offset_word_slice(slice: &[u32]) -> usize {
    slice.as_ptr().align_offset(32)
}
