//@ compile-flags: -Copt-level=3 -Cno-prepopulate-passes -Z merge-functions=disabled
//@ only-x86_64

// Regression test for <https://github.com/rust-lang/rust/issues/157373>.
//
// These cases specifically exercise direct codegen of small non-zero constant
// aggregates as a single integer store. They are chosen so they fail without
// `try_codegen_const_aggregate_as_immediate`.

#![crate_type = "lib"]
#![no_std]

use core::mem::MaybeUninit;

// Inner padding between b (offset 2, size 1) and c (offset 4, size 4).
#[repr(C)]
pub struct InnerPadded {
    a: u16,
    b: u8,
    c: u32,
}

// PR 157690's original ptr::write entry point, checked against the current
// aggregate-immediate codegen shape.
// CHECK-LABEL: @via_ptr_write(
#[no_mangle]
pub fn via_ptr_write(dest: &mut MaybeUninit<InnerPadded>) {
    let val = InnerPadded { a: 0, b: 0, c: 0 };
    // CHECK: %val = alloca [8 x i8], align 4
    // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr %val)
    // CHECK-NEXT: store i64 0, ptr %val, align 4
    // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dest, ptr align 4 %val, i64 8, i1 false)
    unsafe {
        core::ptr::write(dest.as_mut_ptr(), val);
    }
}

// PR 157690's original MaybeUninit::write entry point.
// CHECK-LABEL: @via_maybe_uninit_write(
#[no_mangle]
pub fn via_maybe_uninit_write(dest: &mut MaybeUninit<InnerPadded>) {
    let val = InnerPadded { a: 0, b: 0, c: 0 };
    // CHECK: %val = alloca [8 x i8], align 4
    // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr %val)
    // CHECK-NEXT: store i64 0, ptr %val, align 4
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dest, ptr align 4 %{{.*}}, i64 8, i1 false)
    dest.write(val);
}

// Constant non-zero initialization: emitted as a single store including zero padding.
// CHECK-LABEL: @const_init_non_zero(
#[no_mangle]
pub fn const_init_non_zero(dest: *mut InnerPadded) {
    let val = InnerPadded { a: 0, b: 1, c: 0 };
    // CHECK: %val = alloca [8 x i8], align 4
    // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr %val)
    // CHECK-NEXT: store i64 65536, ptr %val, align 4
    // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dest, ptr align 4 %val, i64 8, i1 false)
    unsafe {
        core::ptr::write(dest, val);
    }
}

// The same direct constant aggregate packing should apply through ptr::write on MaybeUninit.
// CHECK-LABEL: @via_ptr_write_non_zero(
#[no_mangle]
pub fn via_ptr_write_non_zero(dest: &mut MaybeUninit<InnerPadded>) {
    let val = InnerPadded { a: 0, b: 1, c: 0 };
    // CHECK: %val = alloca [8 x i8], align 4
    // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr %val)
    // CHECK-NEXT: store i64 65536, ptr %val, align 4
    // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dest, ptr align 4 %val, i64 8, i1 false)
    unsafe {
        core::ptr::write(dest.as_mut_ptr(), val);
    }
}

// The same direct constant aggregate packing should apply through MaybeUninit::write.
// CHECK-LABEL: @via_maybe_uninit_write_non_zero(
#[no_mangle]
pub fn via_maybe_uninit_write_non_zero(dest: &mut MaybeUninit<InnerPadded>) {
    let val = InnerPadded { a: 0, b: 1, c: 0 };
    // CHECK: %val = alloca [8 x i8], align 4
    // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr %val)
    // CHECK-NEXT: store i64 65536, ptr %val, align 4
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dest, ptr align 4 %{{.*}}, i64 8, i1 false)
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

// PR 157690's trailing-padding-only entry point.
// CHECK-LABEL: @tail_only_write(
#[no_mangle]
pub fn tail_only_write(dest: &mut MaybeUninit<TailOnly>) {
    let val = TailOnly { a: 0, b: 0, c: 0 };
    // CHECK: %val = alloca [8 x i8], align 4
    // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr %val)
    // CHECK-NEXT: store i64 0, ptr %val, align 4
    // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dest, ptr align 4 %val, i64 8, i1 false)
    unsafe {
        core::ptr::write(dest.as_mut_ptr(), val);
    }
}
