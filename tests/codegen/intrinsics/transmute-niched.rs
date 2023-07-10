// revisions: OPT DBG
// [OPT] compile-flags: -C opt-level=3 -C no-prepopulate-passes
// [DBG] compile-flags: -C opt-level=0 -C no-prepopulate-passes
// only-64bit (so I don't need to worry about usize)
// min-llvm-version: 15.0 # this test assumes `ptr`s

#![crate_type = "lib"]

use std::mem::transmute;
use std::num::NonZeroU32;

#[repr(u8)]
pub enum SmallEnum {
    A = 10,
    B = 11,
    C = 12,
}

// CHECK-LABEL: @check_to_enum(
#[no_mangle]
pub unsafe fn check_to_enum(x: i8) -> SmallEnum {
    // OPT: %0 = icmp uge i8 %x, 10
    // OPT: call void @llvm.assume(i1 %0)
    // OPT: %1 = icmp ule i8 %x, 12
    // OPT: call void @llvm.assume(i1 %1)
    // DBG-NOT: icmp
    // DBG-NOT: assume
    // CHECK: ret i8 %x

    transmute(x)
}

// CHECK-LABEL: @check_from_enum(
#[no_mangle]
pub unsafe fn check_from_enum(x: SmallEnum) -> i8 {
    // OPT: %0 = icmp uge i8 %x, 10
    // OPT: call void @llvm.assume(i1 %0)
    // OPT: %1 = icmp ule i8 %x, 12
    // OPT: call void @llvm.assume(i1 %1)
    // DBG-NOT: icmp
    // DBG-NOT: assume
    // CHECK: ret i8 %x

    transmute(x)
}

// CHECK-LABEL: @check_to_ordering(
#[no_mangle]
pub unsafe fn check_to_ordering(x: u8) -> std::cmp::Ordering {
    // OPT: %0 = icmp uge i8 %x, -1
    // OPT: %1 = icmp ule i8 %x, 1
    // OPT: %2 = or i1 %0, %1
    // OPT: call void @llvm.assume(i1 %2)
    // DBG-NOT: icmp
    // DBG-NOT: assume
    // CHECK: ret i8 %x

    transmute(x)
}

// CHECK-LABEL: @check_from_ordering(
#[no_mangle]
pub unsafe fn check_from_ordering(x: std::cmp::Ordering) -> u8 {
    // OPT: %0 = icmp uge i8 %x, -1
    // OPT: %1 = icmp ule i8 %x, 1
    // OPT: %2 = or i1 %0, %1
    // OPT: call void @llvm.assume(i1 %2)
    // DBG-NOT: icmp
    // DBG-NOT: assume
    // CHECK: ret i8 %x

    transmute(x)
}

#[repr(i32)]
pub enum Minus100ToPlus100 {
    A = -100,
    B = -90,
    C = -80,
    D = -70,
    E = -60,
    F = -50,
    G = -40,
    H = -30,
    I = -20,
    J = -10,
    K = 0,
    L = 10,
    M = 20,
    N = 30,
    O = 40,
    P = 50,
    Q = 60,
    R = 70,
    S = 80,
    T = 90,
    U = 100,
}

// CHECK-LABEL: @check_enum_from_char(
#[no_mangle]
pub unsafe fn check_enum_from_char(x: char) -> Minus100ToPlus100 {
    // OPT: %0 = icmp ule i32 %x, 1114111
    // OPT: call void @llvm.assume(i1 %0)
    // OPT: %1 = icmp uge i32 %x, -100
    // OPT: %2 = icmp ule i32 %x, 100
    // OPT: %3 = or i1 %1, %2
    // OPT: call void @llvm.assume(i1 %3)
    // DBG-NOT: icmp
    // DBG-NOT: assume
    // CHECK: ret i32 %x

    transmute(x)
}

// CHECK-LABEL: @check_enum_to_char(
#[no_mangle]
pub unsafe fn check_enum_to_char(x: Minus100ToPlus100) -> char {
    // OPT: %0 = icmp uge i32 %x, -100
    // OPT: %1 = icmp ule i32 %x, 100
    // OPT: %2 = or i1 %0, %1
    // OPT: call void @llvm.assume(i1 %2)
    // OPT: %3 = icmp ule i32 %x, 1114111
    // OPT: call void @llvm.assume(i1 %3)
    // DBG-NOT: icmp
    // DBG-NOT: assume
    // CHECK: ret i32 %x

    transmute(x)
}

// CHECK-LABEL: @check_swap_pair(
#[no_mangle]
pub unsafe fn check_swap_pair(x: (char, NonZeroU32)) -> (NonZeroU32, char) {
    // OPT: %0 = icmp ule i32 %x.0, 1114111
    // OPT: call void @llvm.assume(i1 %0)
    // OPT: %1 = icmp uge i32 %x.0, 1
    // OPT: call void @llvm.assume(i1 %1)
    // OPT: %2 = icmp uge i32 %x.1, 1
    // OPT: call void @llvm.assume(i1 %2)
    // OPT: %3 = icmp ule i32 %x.1, 1114111
    // OPT: call void @llvm.assume(i1 %3)
    // DBG-NOT: icmp
    // DBG-NOT: assume
    // CHECK: %[[P1:.+]] = insertvalue { i32, i32 } poison, i32 %x.0, 0
    // CHECK: %[[P2:.+]] = insertvalue { i32, i32 } %[[P1]], i32 %x.1, 1
    // CHECK: ret { i32, i32 } %[[P2]]

    transmute(x)
}

// CHECK-LABEL: @check_bool_from_ordering(
#[no_mangle]
pub unsafe fn check_bool_from_ordering(x: std::cmp::Ordering) -> bool {
    // OPT: %0 = icmp uge i8 %x, -1
    // OPT: %1 = icmp ule i8 %x, 1
    // OPT: %2 = or i1 %0, %1
    // OPT: call void @llvm.assume(i1 %2)
    // OPT: %3 = icmp ule i8 %x, 1
    // OPT: call void @llvm.assume(i1 %3)
    // DBG-NOT: icmp
    // DBG-NOT: assume
    // CHECK: %[[R:.+]] = trunc i8 %x to i1
    // CHECK: ret i1 %[[R]]

    transmute(x)
}

// CHECK-LABEL: @check_bool_to_ordering(
#[no_mangle]
pub unsafe fn check_bool_to_ordering(x: bool) -> std::cmp::Ordering {
    // CHECK: %_0 = zext i1 %x to i8
    // OPT: %0 = icmp ule i8 %_0, 1
    // OPT: call void @llvm.assume(i1 %0)
    // OPT: %1 = icmp uge i8 %_0, -1
    // OPT: %2 = icmp ule i8 %_0, 1
    // OPT: %3 = or i1 %1, %2
    // OPT: call void @llvm.assume(i1 %3)
    // DBG-NOT: icmp
    // DBG-NOT: assume
    // CHECK: ret i8 %_0

    transmute(x)
}
