//@ compile-flags: -Copt-level=0 -Cno-prepopulate-passes
//@ only-64bit

#![crate_type = "lib"]

use std::cmp::Ordering;
use std::num::NonZero;
use std::ptr::NonNull;

#[no_mangle]
fn make_some_bool(x: bool) -> Option<bool> {
    // CHECK-LABEL: i8 @make_some_bool(i1 zeroext %x)
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[WIDER:.+]] = zext i1 %x to i8
    // CHECK-NEXT: ret i8 %[[WIDER]]
    Some(x)
}

#[no_mangle]
fn make_none_bool() -> Option<bool> {
    // CHECK-LABEL: i8 @make_none_bool()
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret i8 2
    None
}

#[no_mangle]
fn make_some_ordering(x: Ordering) -> Option<Ordering> {
    // CHECK-LABEL: i8 @make_some_ordering(i8{{( signext)?}} %x)
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret i8 %x
    Some(x)
}

#[no_mangle]
fn make_some_u16(x: u16) -> Option<u16> {
    // CHECK-LABEL: { i16, i16 } @make_some_u16(i16{{( zeroext)?}} %x)
    // CHECK-NEXT: start:
    // CHECK-NEXT: %0 = insertvalue { i16, i16 } { i16 1, i16 poison }, i16 %x, 1
    // CHECK-NEXT: ret { i16, i16 } %0
    Some(x)
}

#[no_mangle]
fn make_none_u16() -> Option<u16> {
    // CHECK-LABEL: { i16, i16 } @make_none_u16()
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret { i16, i16 } { i16 0, i16 undef }
    None
}

#[no_mangle]
fn make_some_nzu32(x: NonZero<u32>) -> Option<NonZero<u32>> {
    // CHECK-LABEL: i32 @make_some_nzu32(i32{{( signext)?}} %x)
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret i32 %x
    Some(x)
}

#[no_mangle]
fn make_ok_ptr(x: NonNull<u16>) -> Result<NonNull<u16>, usize> {
    // CHECK-LABEL: { i64, ptr } @make_ok_ptr(ptr %x)
    // CHECK-NEXT: start:
    // CHECK-NEXT: %0 = insertvalue { i64, ptr } { i64 0, ptr poison }, ptr %x, 1
    // CHECK-NEXT: ret { i64, ptr } %0
    Ok(x)
}

#[no_mangle]
fn make_ok_int(x: usize) -> Result<usize, NonNull<u16>> {
    // CHECK-LABEL: { i64, ptr } @make_ok_int(i64 %x)
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[NOPROV:.+]] = getelementptr i8, ptr null, i64 %x
    // CHECK-NEXT: %[[R:.+]] = insertvalue { i64, ptr } { i64 0, ptr poison }, ptr %[[NOPROV]], 1
    // CHECK-NEXT: ret { i64, ptr } %[[R]]
    Ok(x)
}

#[no_mangle]
fn make_some_ref(x: &u16) -> Option<&u16> {
    // CHECK-LABEL: ptr @make_some_ref(ptr align 2 %x)
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret ptr %x
    Some(x)
}

#[no_mangle]
fn make_none_ref<'a>() -> Option<&'a u16> {
    // CHECK-LABEL: ptr @make_none_ref()
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret ptr null
    None
}

#[inline(never)]
fn make_err_generic<E>(e: E) -> Result<u32, E> {
    // CHECK-LABEL: define{{.+}}make_err_generic
    // CHECK-NEXT: start:
    // CHECK-NEXT: call void @llvm.trap()
    // CHECK-NEXT: ret i32 poison
    Err(e)
}

#[no_mangle]
fn make_uninhabited_err_indirectly(n: Never) -> Result<u32, Never> {
    // CHECK-LABEL: i32 @make_uninhabited_err_indirectly()
    // CHECK-NEXT: start:
    // CHECK-NEXT: call{{.+}}make_err_generic
    make_err_generic(n)
}

#[no_mangle]
fn make_fully_uninhabited_result(v: u32, n: Never) -> Result<(u32, Never), (Never, u32)> {
    // Actually reaching this would be UB, so we don't actually build a result.

    // CHECK-LABEL: { i32, i32 } @make_fully_uninhabited_result(i32{{( signext)?}} %v)
    // CHECK-NEXT: start:
    // CHECK-NEXT: call void @llvm.trap()
    // CHECK-NEXT: call void @llvm.trap()
    // CHECK-NEXT: call void @llvm.trap()
    // CHECK-NEXT: unreachable
    Ok((v, n))
}

enum Never {}
