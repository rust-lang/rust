//@ add-minicore
//@ compile-flags: -Copt-level=3 -Cno-prepopulate-passes -Z merge-functions=disabled -Z randomize-layout=no
//@ revisions: powerpc64 x86_64
//@[powerpc64] compile-flags: --target powerpc64-unknown-linux-gnu
//@[powerpc64] needs-llvm-components: powerpc
//@[x86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@[x86_64] needs-llvm-components: x86

// Regression test for <https://github.com/rust-lang/rust/issues/157373>.
//
// These cases specifically exercise direct codegen of small non-zero constant
// aggregates as a single integer store. They are chosen so they fail without
// `try_codegen_const_aggregate_as_immediate`.

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;

use minicore::*;

#[inline(always)]
unsafe fn ptr_write<T>(dest: *mut T, value: T) {
    *dest = value;
}

trait MaybeUninitExt<T> {
    fn as_mut_ptr(&mut self) -> *mut T;
    fn write(&mut self, value: T);
}

impl<T> MaybeUninitExt<T> for MaybeUninit<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self as *mut _ as *mut T
    }

    fn write(&mut self, value: T) {
        unsafe {
            ptr_write(self.as_mut_ptr(), value);
        }
    }
}

// Inner padding between b (offset 2, size 1) and c (offset 4, size 4).
#[repr(C)]
pub struct InnerPadded {
    a: u16,
    b: u8,
    c: u32,
}

#[repr(transparent)]
pub struct Nested1(InnerPadded);

#[repr(transparent)]
pub struct Nested2(Nested1);

// PR 157690's original ptr::write entry point, checked against the current
// aggregate-immediate codegen shape.
// CHECK-LABEL: @via_ptr_write(
#[no_mangle]
pub fn via_ptr_write(dest: &mut MaybeUninit<InnerPadded>) {
    let val = InnerPadded { a: 0, b: 0, c: 0 };
    // CHECK: %val = alloca [8 x i8], align 4
    // CHECK-NEXT: call void @llvm.lifetime.start.p0({{(i64 8, )?}}ptr %val)
    // CHECK-NEXT: store i64 0, ptr %val, align 4
    // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dest, ptr align 4 %val, i64 8, i1 false)
    unsafe {
        ptr_write(dest.as_mut_ptr(), val);
    }
}

// PR 157690's original MaybeUninit::write entry point.
// CHECK-LABEL: @via_maybe_uninit_write(
#[no_mangle]
pub fn via_maybe_uninit_write(dest: &mut MaybeUninit<InnerPadded>) {
    let val = InnerPadded { a: 0, b: 0, c: 0 };
    // CHECK: %val = alloca [8 x i8], align 4
    // CHECK-NEXT: call void @llvm.lifetime.start.p0({{(i64 8, )?}}ptr %val)
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
    // CHECK-NEXT: call void @llvm.lifetime.start.p0({{(i64 8, )?}}ptr %val)
    // x86_64-NEXT: store i64 65536, ptr %val, align 4
    // powerpc64-NEXT: store i64 1099511627776, ptr %val, align 4
    // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dest, ptr align 4 %val, i64 8, i1 false)
    unsafe {
        ptr_write(dest, val);
    }
}

// From issue #157373: nesting wrapper structs used to change the lowering
// shape enough that LLVM would sometimes find the wide store only in the
// nested case.
// CHECK-LABEL: @bad(
#[no_mangle]
pub fn bad(a: &mut InnerPadded) {
    let x = InnerPadded { a: 0, b: 1, c: 0 };
    // x86_64: store i64 65536, ptr %x, align 4
    // powerpc64: store i64 1099511627776, ptr %x, align 4
    *a = x;
}

// CHECK-LABEL: @good(
#[no_mangle]
pub fn good(a: &mut Nested2) {
    let x = InnerPadded { a: 0, b: 1, c: 0 };
    // x86_64: store i64 65536, ptr %x, align 4
    // powerpc64: store i64 1099511627776, ptr %x, align 4
    *a = Nested2(Nested1(x));
}

// The same direct constant aggregate packing should apply through ptr::write on MaybeUninit.
// CHECK-LABEL: @via_ptr_write_non_zero(
#[no_mangle]
pub fn via_ptr_write_non_zero(dest: &mut MaybeUninit<InnerPadded>) {
    let val = InnerPadded { a: 0, b: 1, c: 0 };
    // CHECK: %val = alloca [8 x i8], align 4
    // CHECK-NEXT: call void @llvm.lifetime.start.p0({{(i64 8, )?}}ptr %val)
    // x86_64-NEXT: store i64 65536, ptr %val, align 4
    // powerpc64-NEXT: store i64 1099511627776, ptr %val, align 4
    // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dest, ptr align 4 %val, i64 8, i1 false)
    unsafe {
        ptr_write(dest.as_mut_ptr(), val);
    }
}

// The same direct constant aggregate packing should apply through MaybeUninit::write.
// CHECK-LABEL: @via_maybe_uninit_write_non_zero(
#[no_mangle]
pub fn via_maybe_uninit_write_non_zero(dest: &mut MaybeUninit<InnerPadded>) {
    let val = InnerPadded { a: 0, b: 1, c: 0 };
    // CHECK: %val = alloca [8 x i8], align 4
    // CHECK-NEXT: call void @llvm.lifetime.start.p0({{(i64 8, )?}}ptr %val)
    // x86_64-NEXT: store i64 65536, ptr %val, align 4
    // powerpc64-NEXT: store i64 1099511627776, ptr %val, align 4
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dest, ptr align 4 %{{.*}}, i64 8, i1 false)
    dest.write(val);
}

// CHECK-LABEL: @bad_non_zero(
#[no_mangle]
pub fn bad_non_zero(a: &mut InnerPadded) {
    let x = InnerPadded { a: 0, b: 1, c: 0 };
    // CHECK: %x = alloca [8 x i8], align 4
    // CHECK-NEXT: call void @llvm.lifetime.start.p0({{(i64 8, )?}}ptr %x)
    // x86_64-NEXT: store i64 65536, ptr %x, align 4
    // powerpc64-NEXT: store i64 1099511627776, ptr %x, align 4
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a, ptr align 4 %x, i64 8, i1 false)
    *a = x;
}

// CHECK-LABEL: @good_non_zero(
#[no_mangle]
pub fn good_non_zero(a: &mut Nested2) {
    let x = InnerPadded { a: 0, b: 1, c: 0 };
    // CHECK: %x = alloca [8 x i8], align 4
    // CHECK-NEXT: call void @llvm.lifetime.start.p0({{(i64 8, )?}}ptr %x)
    // x86_64-NEXT: store i64 65536, ptr %x, align 4
    // powerpc64-NEXT: store i64 1099511627776, ptr %x, align 4
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a, ptr align 4 %{{.*}}, i64 8, i1 false)
    *a = Nested2(Nested1(x));
}

// Trailing padding only (no inter-field padding): a (offset 0, size 4),
// b (offset 4, size 2), c (offset 6, size 1), trailing pad (offset 7, size 1).
#[repr(C)]
pub struct TailOnly {
    a: u32,
    b: u16,
    c: u8,
}

type TupleTailOnly = (u32, u16, u8);

// PR 157690's trailing-padding-only entry point.
// CHECK-LABEL: @tail_only_write(
#[no_mangle]
pub fn tail_only_write(dest: &mut MaybeUninit<TailOnly>) {
    let val = TailOnly { a: 0, b: 0, c: 0 };
    // CHECK: %val = alloca [8 x i8], align 4
    // CHECK-NEXT: call void @llvm.lifetime.start.p0({{(i64 8, )?}}ptr %val)
    // CHECK-NEXT: store i64 0, ptr %val, align 4
    // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dest, ptr align 4 %val, i64 8, i1 false)
    unsafe {
        ptr_write(dest.as_mut_ptr(), val);
    }
}

// Tuple aggregates should use the same const-packing path as structs when the
// whole tuple is constant and small enough to fit in an integer store.
// CHECK-LABEL: @tuple_tail_only_non_zero(
#[no_mangle]
pub fn tuple_tail_only_non_zero(dest: *mut TupleTailOnly) {
    let val: TupleTailOnly = (0, 1, 0);
    // CHECK: %val = alloca [8 x i8], align 4
    // CHECK-NEXT: call void @llvm.lifetime.start.p0({{(i64 8, )?}}ptr %val)
    // x86_64-NEXT: store i64 4294967296, ptr %val, align 4
    // powerpc64-NEXT: store i64 65536, ptr %val, align 4
    // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dest, ptr align 4 %val, i64 8, i1 false)
    unsafe {
        ptr_write(dest, val);
    }
}

// Regression test for the debug assertion failure in
// `try_codegen_const_aggregate_as_immediate` when the MIR aggregate's
// variant index doesn't match the layout's `Variants::Single { index }`.
//
// When `Data(Void)` is uninhabited, the layout of `E<Void>` collapses to
// `Variants::Single { index: 0 }` (only `Empty`). But generic code
// monomorphized with `T = Void` still contains an aggregate for `Data(x)`
// with 1 operand. The optimization must bail out gracefully instead of
// asserting `operands.len() == dest.layout.fields.count()` (1 == 0).
//
// See <https://github.com/rust-lang/rust/pull/157690>.
enum Void {}

enum E<T> {
    Empty,
    Data(T),
}

#[inline(never)]
fn make_data<T>(x: T) -> E<T> {
    E::Data(x)
}

// Force codegen of `make_data::<Void>`. Without the variant index check,
// this triggers: assertion `left == right` failed (left: 1, right: 0).
// CHECK-LABEL: @force_variant_mismatch(
#[no_mangle]
pub fn force_variant_mismatch() -> fn(Void) -> E<Void> {
    make_data::<Void>
}
