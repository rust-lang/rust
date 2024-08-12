//@ compile-flags: -C opt-level=3 -C no-prepopulate-passes
//@ min-llvm-version: 19
#![crate_type = "lib"]

use std::mem::transmute;
use std::num::NonZero;

#[repr(u8)]
pub enum SmallEnum {
    A = 10,
    B = 11,
    C = 12,
}

// CHECK: noundef range(i8 10, 13) i8 @check_to_enum(
#[no_mangle]
pub unsafe fn check_to_enum(x: i8) -> SmallEnum {
    transmute(x)
}

// CHECK: @check_from_enum(i8 noundef range(i8 10, 13) %x)
#[no_mangle]
pub unsafe fn check_from_enum(x: SmallEnum) -> i8 {
    transmute(x)
}

// CHECK: noundef range(i8 -1, 2) i8 @check_to_ordering(
#[no_mangle]
pub unsafe fn check_to_ordering(x: u8) -> std::cmp::Ordering {
    transmute(x)
}

// CHECK: @check_from_ordering(i8 noundef range(i8 -1, 2) %x)
#[no_mangle]
pub unsafe fn check_from_ordering(x: std::cmp::Ordering) -> u8 {
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

// CHECK: noundef range(i32 -100, 101) i32 @check_enum_from_char(i32 noundef range(i32 0, 1114112) %x)
#[no_mangle]
pub unsafe fn check_enum_from_char(x: char) -> Minus100ToPlus100 {
    transmute(x)
}

// CHECK: noundef range(i32 0, 1114112) i32 @check_enum_to_char(i32 noundef range(i32 -100, 101) %x)
#[no_mangle]
pub unsafe fn check_enum_to_char(x: Minus100ToPlus100) -> char {
    transmute(x)
}

// CHECK: @check_swap_pair(i32 noundef range(i32 0, 1114112) %x.0, i32 noundef range(i32 1, 0) %x.1)
#[no_mangle]
pub unsafe fn check_swap_pair(x: (char, NonZero<u32>)) -> (NonZero<u32>, char) {
    transmute(x)
}

// CHECK: @check_bool_from_ordering(i8 noundef range(i8 -1, 2) %x)
#[no_mangle]
pub unsafe fn check_bool_from_ordering(x: std::cmp::Ordering) -> bool {
    transmute(x)
}

// CHECK: noundef range(i8 -1, 2) i8 @check_bool_to_ordering(
#[no_mangle]
pub unsafe fn check_bool_to_ordering(x: bool) -> std::cmp::Ordering {
    transmute(x)
}
