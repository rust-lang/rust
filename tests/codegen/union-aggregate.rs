//@ compile-flags: -Copt-level=0 -Cno-prepopulate-passes
//@ min-llvm-version: 19
//@ only-64bit

#![crate_type = "lib"]
#![feature(transparent_unions)]

#[repr(transparent)]
union MU<T: Copy> {
    uninit: (),
    value: T,
}

use std::cmp::Ordering;
use std::num::NonZero;
use std::ptr::NonNull;

#[no_mangle]
fn make_mu_bool(x: bool) -> MU<bool> {
    // CHECK-LABEL: i8 @make_mu_bool(i1 zeroext %x)
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[WIDER:.+]] = zext i1 %x to i8
    // CHECK-NEXT: ret i8 %[[WIDER]]
    MU { value: x }
}

#[no_mangle]
fn make_mu_bool_uninit() -> MU<bool> {
    // CHECK-LABEL: i8 @make_mu_bool_uninit()
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret i8 undef
    MU { uninit: () }
}

#[no_mangle]
fn make_mu_ref(x: &u16) -> MU<&u16> {
    // CHECK-LABEL: ptr @make_mu_ref(ptr align 2 %x)
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret ptr %x
    MU { value: x }
}

#[no_mangle]
fn make_mu_ref_uninit<'a>() -> MU<&'a u16> {
    // CHECK-LABEL: ptr @make_mu_ref_uninit()
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret ptr undef
    MU { uninit: () }
}

#[no_mangle]
fn make_mu_str(x: &str) -> MU<&str> {
    // CHECK-LABEL: { ptr, i64 } @make_mu_str(ptr align 1 %x.0, i64 %x.1)
    // CHECK-NEXT: start:
    // CHECK-NEXT: %0 = insertvalue { ptr, i64 } poison, ptr %x.0, 0
    // CHECK-NEXT: %1 = insertvalue { ptr, i64 } %0, i64 %x.1, 1
    // CHECK-NEXT: ret { ptr, i64 } %1
    MU { value: x }
}

#[no_mangle]
fn make_mu_str_uninit<'a>() -> MU<&'a str> {
    // CHECK-LABEL: { ptr, i64 } @make_mu_str_uninit()
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret { ptr, i64 } undef
    MU { uninit: () }
}

#[no_mangle]
fn make_mu_pair(x: (u8, u32)) -> MU<(u8, u32)> {
    // CHECK-LABEL: { i8, i32 } @make_mu_pair(i8 %x.0, i32 %x.1)
    // CHECK-NEXT: start:
    // CHECK-NEXT: %0 = insertvalue { i8, i32 } poison, i8 %x.0, 0
    // CHECK-NEXT: %1 = insertvalue { i8, i32 } %0, i32 %x.1, 1
    // CHECK-NEXT: ret { i8, i32 } %1
    MU { value: x }
}

#[no_mangle]
fn make_mu_pair_uninit() -> MU<(u8, u32)> {
    // CHECK-LABEL: { i8, i32 } @make_mu_pair_uninit()
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret { i8, i32 } undef
    MU { uninit: () }
}
