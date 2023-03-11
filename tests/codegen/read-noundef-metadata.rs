// compile-flags: -O -Z merge-functions=disabled
// no-system-llvm
// ignore-debug (the extra assertions get in the way)

#![crate_type = "lib"]

// Ensure that various forms of reading pointers correctly annotate the `load`s
// with `!noundef` metadata to enable extra optimization.  The functions return
// `MaybeUninit` to keep it from being inferred from the function type.

use std::mem::MaybeUninit;

// CHECK-LABEL: define i8 @copy_byte(
#[no_mangle]
pub unsafe fn copy_byte(p: *const u8) -> MaybeUninit<u8> {
    // CHECK-NOT: load
    // CHECK: load i8, ptr %p, align 1
    // CHECK-SAME: !noundef !
    // CHECK-NOT: load
    MaybeUninit::new(*p)
}

// CHECK-LABEL: define i8 @read_byte(
#[no_mangle]
pub unsafe fn read_byte(p: *const u8) -> MaybeUninit<u8> {
    // CHECK-NOT: load
    // CHECK: load i8, ptr %p, align 1
    // CHECK-SAME: !noundef !
    // CHECK-NOT: load
    MaybeUninit::new(p.read())
}

// CHECK-LABEL: define i8 @read_byte_maybe_uninit(
#[no_mangle]
pub unsafe fn read_byte_maybe_uninit(p: *const MaybeUninit<u8>) -> MaybeUninit<u8> {
    // CHECK-NOT: load
    // CHECK: load i8, ptr %p, align 1
    // CHECK-NOT: noundef
    // CHECK-NOT: load
    p.read()
}

// CHECK-LABEL: define i8 @read_byte_assume_init(
#[no_mangle]
pub unsafe fn read_byte_assume_init(p: &MaybeUninit<u8>) -> MaybeUninit<u8> {
    // CHECK-NOT: load
    // CHECK: load i8, ptr %p, align 1
    // CHECK-SAME: !noundef !
    // CHECK-NOT: load
    MaybeUninit::new(p.assume_init_read())
}
