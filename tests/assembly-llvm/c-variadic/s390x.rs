//@ add-minicore
//@ assembly-output: emit-asm
//
//@ compile-flags: -Copt-level=3 --target s390x-unknown-linux-gnu
//@ needs-llvm-components: systemz
#![feature(c_variadic, no_core, lang_items, intrinsics, rustc_attrs)]
#![no_core]
#![crate_type = "lib"]

// Check that the assembly that rustc generates matches what clang emits.

extern crate minicore;
use minicore::*;

#[lang = "va_arg_safe"]
pub unsafe trait VaArgSafe {}

unsafe impl VaArgSafe for i32 {}
unsafe impl VaArgSafe for i64 {}
unsafe impl VaArgSafe for i128 {}
unsafe impl VaArgSafe for f64 {}
unsafe impl<T> VaArgSafe for *const T {}

#[repr(transparent)]
struct VaListInner {
    ptr: *const c_void,
}

#[repr(transparent)]
#[lang = "va_list"]
pub struct VaList<'a> {
    inner: VaListInner,
    _marker: PhantomData<&'a mut ()>,
}

#[rustc_intrinsic]
#[rustc_nounwind]
pub const unsafe fn va_arg<T: VaArgSafe>(ap: &mut VaList<'_>) -> T;

#[unsafe(no_mangle)]
unsafe extern "C" fn read_f64(ap: &mut VaList<'_>) -> f64 {
    // CHECK-LABEL: read_f64:
    // CHECK: lg %r3, 8(%r2)
    // CHECK-NEXT: clgijh %r3, 3, .LBB0_2
    // CHECK-NEXT: lg %r1, 24(%r2)
    // CHECK-NEXT: sllg %r4, %r3, 3
    // CHECK-NEXT: la %r1, 128(%r4,%r1)
    // CHECK-NEXT: la %r0, 1(%r3)
    // CHECK-NEXT: stg %r0, 8(%r2)
    // CHECK-NEXT: ld %f0, 0(%r1)
    // CHECK-NEXT: br %r14
    // CHECK-NEXT: .LBB0_2:
    // CHECK-NEXT: lg %r1, 16(%r2)
    // CHECK-NEXT: la %r0, 8(%r1)
    // CHECK-NEXT: stg %r0, 16(%r2)
    // CHECK-NEXT: ld %f0, 0(%r1)
    // CHECK-NEXT: br %r14
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i32(ap: &mut VaList<'_>) -> i32 {
    // CHECK-LABEL: read_i32:
    // CHECK: lg %r3, 0(%r2)
    // CHECK-NEXT: clgijh %r3, 4, .LBB1_2
    // CHECK-NEXT: lg %r1, 24(%r2)
    // CHECK-NEXT: sllg %r4, %r3, 3
    // CHECK-NEXT: la %r1, 20(%r4,%r1)
    // CHECK-NEXT: la %r0, 1(%r3)
    // CHECK-NEXT: stg %r0, 0(%r2)
    // CHECK-NEXT: lgf %r2, 0(%r1)
    // CHECK-NEXT: br %r14
    // CHECK-NEXT: .LBB1_2:
    // CHECK-NEXT: lg %r3, 16(%r2)
    // CHECK-NEXT: la %r1, 4(%r3)
    // CHECK-NEXT: la %r0, 8(%r3)
    // CHECK-NEXT: stg %r0, 16(%r2)
    // CHECK-NEXT: lgf %r2, 0(%r1)
    // CHECK-NEXT: br %r14
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i64(ap: &mut VaList<'_>) -> i64 {
    // CHECK-LABEL: read_i64:
    // CHECK: lg %r3, 0(%r2)
    // CHECK-NEXT: clgijh %r3, 4, .LBB2_2
    // CHECK-NEXT: lg %r1, 24(%r2)
    // CHECK-NEXT: sllg %r4, %r3, 3
    // CHECK-NEXT: la %r1, 16(%r4,%r1)
    // CHECK-NEXT: la %r0, 1(%r3)
    // CHECK-NEXT: stg %r0, 0(%r2)
    // CHECK-NEXT: lg %r2, 0(%r1)
    // CHECK-NEXT: br %r14
    // CHECK-NEXT: .LBB2_2:
    // CHECK-NEXT: lg %r1, 16(%r2)
    // CHECK-NEXT: la %r0, 8(%r1)
    // CHECK-NEXT: stg %r0, 16(%r2)
    // CHECK-NEXT: lg %r2, 0(%r1)
    // CHECK-NEXT: br %r14
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i128(ap: &mut VaList<'_>) -> i128 {
    // CHECK-LABEL: read_i128:
    // CHECK: lg %r4, 0(%r3)
    // CHECK-NEXT: clgijh %r4, 4, .LBB3_2
    // CHECK-NEXT: lg %r1, 24(%r3)
    // CHECK-NEXT: sllg %r5, %r4, 3
    // CHECK-NEXT: la %r1, 16(%r5,%r1)
    // CHECK-NEXT: la %r0, 1(%r4)
    // CHECK-NEXT: stg %r0, 0(%r3)
    // CHECK-NEXT: lg %r1, 0(%r1)
    // CHECK-NEXT: mvc 8(8,%r2), 8(%r1)
    // CHECK-NEXT: mvc 0(8,%r2), 0(%r1)
    // CHECK-NEXT: br %r14
    // CHECK-NEXT: .LBB3_2:
    // CHECK-NEXT: lg %r1, 16(%r3)
    // CHECK-NEXT: la %r0, 8(%r1)
    // CHECK-NEXT: stg %r0, 16(%r3)
    // CHECK-NEXT: lg %r1, 0(%r1)
    // CHECK-NEXT: mvc 8(8,%r2), 8(%r1)
    // CHECK-NEXT: mvc 0(8,%r2), 0(%r1)
    // CHECK-NEXT: br %r14
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_ptr(ap: &mut VaList<'_>) -> *const u8 {
    // CHECK: read_ptr = read_i64
    va_arg(ap)
}
