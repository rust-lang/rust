//@ compile-flags: -Copt-level=3 -Z merge-functions=disabled
//@ only-x86_64

// Regression test for <https://github.com/rust-lang/rust/issues/157373>.
//
// When a struct with inner padding is copied via `typed_place_copy` (i.e.,
// memcpy), the compiler now zeroes the padding gaps in the destination. This
// lets LLVM merge stores through the padding gap, producing wider (and fewer)
// stores instead of individual field-level stores.

#![crate_type = "lib"]

use std::mem::MaybeUninit;

// Inner padding between b (offset 2, size 1) and c (offset 4, size 4).
#[repr(C)]
pub struct InnerPadded {
    a: u16,
    b: u8,
    c: u32,
}

// CHECK-LABEL: @via_ptr_write(
#[no_mangle]
pub fn via_ptr_write(dest: &mut MaybeUninit<InnerPadded>) {
    let val = InnerPadded { a: 0, b: 0, c: 0 };
    // CHECK: store i64 0, ptr %dest, align 4
    // CHECK-NEXT: ret void
    unsafe {
        std::ptr::write(dest.as_mut_ptr(), val);
    }
}

// CHECK-LABEL: @via_maybe_uninit_write(
#[no_mangle]
pub fn via_maybe_uninit_write(dest: &mut MaybeUninit<InnerPadded>) {
    let val = InnerPadded { a: 0, b: 0, c: 0 };
    // CHECK: store i64 0, ptr %dest, align 4
    // CHECK-NEXT: ret void
    dest.write(val);
}

// Trailing padding only (no inter-field padding): a (offset 0, size 4),
// b (offset 4, size 2), c (offset 6, size 1), trailing pad (offset 7, size 1).
#[repr(C)]
pub struct TailOnly {
    a: u32,
    b: u16,
    c: u8,
}

// CHECK-LABEL: @tail_only_write(
#[no_mangle]
pub fn tail_only_write(dest: &mut MaybeUninit<TailOnly>) {
    let val = TailOnly { a: 0, b: 0, c: 0 };
    // CHECK: store i64 0, ptr %dest, align 4
    // CHECK-NEXT: ret void
    unsafe {
        std::ptr::write(dest.as_mut_ptr(), val);
    }
}
