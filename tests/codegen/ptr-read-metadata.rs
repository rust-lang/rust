// compile-flags: -O -Z merge-functions=disabled
// no-system-llvm
// ignore-debug (the extra assertions get in the way)

#![crate_type = "lib"]

// Ensure that various forms of reading pointers correctly annotate the `load`s
// with `!noundef` and `!range` metadata to enable extra optimization.

use std::mem::MaybeUninit;

// CHECK-LABEL: define noundef i8 @copy_byte(
#[no_mangle]
pub unsafe fn copy_byte(p: *const u8) -> u8 {
    // CHECK-NOT: load
    // CHECK: load i8, ptr %p, align 1
    // CHECK-SAME: !noundef !
    // CHECK-NOT: load
    *p
}

// CHECK-LABEL: define noundef i8 @read_byte(
#[no_mangle]
pub unsafe fn read_byte(p: *const u8) -> u8 {
    // CHECK-NOT: load
    // CHECK: load i8, ptr %p, align 1
    // CHECK-SAME: !noundef !
    // CHECK-NOT: load
    p.read()
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

// CHECK-LABEL: define noundef i8 @read_byte_assume_init(
#[no_mangle]
pub unsafe fn read_byte_assume_init(p: &MaybeUninit<u8>) -> u8 {
    // CHECK-NOT: load
    // CHECK: load i8, ptr %p, align 1
    // CHECK-SAME: !noundef !
    // CHECK-NOT: load
    p.assume_init_read()
}

// CHECK-LABEL: define noundef i32 @copy_char(
#[no_mangle]
pub unsafe fn copy_char(p: *const char) -> char {
    // CHECK-NOT: load
    // CHECK: load i32, ptr %p
    // CHECK-SAME: !range ![[RANGE:[0-9]+]]
    // CHECK-SAME: !noundef !
    // CHECK-NOT: load
    *p
}

// CHECK-LABEL: define noundef i32 @read_char(
#[no_mangle]
pub unsafe fn read_char(p: *const char) -> char {
    // CHECK-NOT: load
    // CHECK: load i32, ptr %p
    // CHECK-SAME: !range ![[RANGE]]
    // CHECK-SAME: !noundef !
    // CHECK-NOT: load
    p.read()
}

// CHECK-LABEL: define i32 @read_char_maybe_uninit(
#[no_mangle]
pub unsafe fn read_char_maybe_uninit(p: *const MaybeUninit<char>) -> MaybeUninit<char> {
    // CHECK-NOT: load
    // CHECK: load i32, ptr %p
    // CHECK-NOT: range
    // CHECK-NOT: noundef
    // CHECK-NOT: load
    p.read()
}

// CHECK-LABEL: define noundef i32 @read_char_assume_init(
#[no_mangle]
pub unsafe fn read_char_assume_init(p: &MaybeUninit<char>) -> char {
    // CHECK-NOT: load
    // CHECK: load i32, ptr %p
    // CHECK-SAME: !range ![[RANGE]]
    // CHECK-SAME: !noundef !
    // CHECK-NOT: load
    p.assume_init_read()
}

// CHECK: ![[RANGE]] = !{i32 0, i32 1114112}
