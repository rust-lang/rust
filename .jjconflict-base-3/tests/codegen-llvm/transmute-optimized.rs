//@ compile-flags: -Copt-level=3 -Z merge-functions=disabled
#![crate_type = "lib"]

// This tests that LLVM can optimize based on the niches in the source or
// destination types for transmutes.

#[repr(u32)]
pub enum AlwaysZero32 {
    X = 0,
}

// CHECK-LABEL: i32 @issue_109958(i32
#[no_mangle]
pub fn issue_109958(x: AlwaysZero32) -> i32 {
    // CHECK: ret i32 0
    unsafe { std::mem::transmute(x) }
}

// CHECK-LABEL: i1 @reference_is_null(ptr
#[no_mangle]
pub fn reference_is_null(x: &i32) -> bool {
    // CHECK: ret i1 false
    let p: *const i32 = unsafe { std::mem::transmute(x) };
    p.is_null()
}

// CHECK-LABEL: i1 @non_null_is_null(ptr
#[no_mangle]
pub fn non_null_is_null(x: std::ptr::NonNull<i32>) -> bool {
    // CHECK: ret i1 false
    let p: *const i32 = unsafe { std::mem::transmute(x) };
    p.is_null()
}

// CHECK-LABEL: i1 @non_zero_is_null(
#[no_mangle]
pub fn non_zero_is_null(x: std::num::NonZero<usize>) -> bool {
    // CHECK: ret i1 false
    let p: *const i32 = unsafe { std::mem::transmute(x) };
    p.is_null()
}

// CHECK-LABEL: i1 @non_null_is_zero(ptr
#[no_mangle]
pub fn non_null_is_zero(x: std::ptr::NonNull<i32>) -> bool {
    // CHECK: ret i1 false
    let a: isize = unsafe { std::mem::transmute(x) };
    a == 0
}

// CHECK-LABEL: i1 @bool_ordering_is_ge(i1
#[no_mangle]
pub fn bool_ordering_is_ge(x: bool) -> bool {
    // CHECK: ret i1 true
    let y: std::cmp::Ordering = unsafe { std::mem::transmute(x) };
    y.is_ge()
}

// CHECK-LABEL: i1 @ordering_is_ge_then_transmute_to_bool(i8
#[no_mangle]
pub fn ordering_is_ge_then_transmute_to_bool(x: std::cmp::Ordering) -> bool {
    let r = x.is_ge();
    let _: bool = unsafe { std::mem::transmute(x) };
    r
}

// CHECK-LABEL: i32 @normal_div(i32
#[no_mangle]
pub fn normal_div(a: u32, b: u32) -> u32 {
    // CHECK: call core::panicking::panic
    a / b
}

// CHECK-LABEL: i32 @div_transmute_nonzero(i32
#[no_mangle]
pub fn div_transmute_nonzero(a: u32, b: std::num::NonZero<i32>) -> u32 {
    // CHECK-NOT: call core::panicking::panic
    // CHECK: %[[R:.+]] = udiv i32 %a, %b
    // CHECK-NEXT: ret i32 %[[R]]
    // CHECK-NOT: call core::panicking::panic
    let d: u32 = unsafe { std::mem::transmute(b) };
    a / d
}

#[repr(i8)]
pub enum OneTwoThree {
    One = 1,
    Two = 2,
    Three = 3,
}

// CHECK-LABEL: i8 @ordering_transmute_onetwothree(i8
#[no_mangle]
pub unsafe fn ordering_transmute_onetwothree(x: std::cmp::Ordering) -> OneTwoThree {
    // CHECK: ret i8 1
    std::mem::transmute(x)
}

// CHECK-LABEL: i8 @onetwothree_transmute_ordering(i8
#[no_mangle]
pub unsafe fn onetwothree_transmute_ordering(x: OneTwoThree) -> std::cmp::Ordering {
    // CHECK: ret i8 1
    std::mem::transmute(x)
}

// CHECK-LABEL: i1 @char_is_negative(i32
#[no_mangle]
pub fn char_is_negative(c: char) -> bool {
    // CHECK: ret i1 false
    let x: i32 = unsafe { std::mem::transmute(c) };
    x < 0
}

// CHECK-LABEL: i1 @transmute_to_char_is_negative(i32
#[no_mangle]
pub fn transmute_to_char_is_negative(x: i32) -> bool {
    // CHECK: ret i1 false
    let _c: char = unsafe { std::mem::transmute(x) };
    x < 0
}
