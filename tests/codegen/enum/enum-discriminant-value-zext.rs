//@ compile-flags: -Cno-prepopulate-passes -O
//@ min-llvm-version: 18

// Test for adding `nneg` to non-negative when using `zext`.

#![crate_type = "lib"]

pub enum Enum0 {
    A = 0,
    B = 127, // 127i8
}

pub enum Enum1 {
    A = 0,
    B = 128, // -128i8
}

pub enum Enum2 {
    A = 0,
    B = 255, // -1i8
}

// CHECK-LABEL: @discriminant_0
#[no_mangle]
pub fn discriminant_0(e: Enum0) -> isize {
    // CHECK: zext nneg i8
    e as isize
}

// CHECK-LABEL: @discriminant_1
#[no_mangle]
pub fn discriminant_1(e: Enum1) -> isize {
    // CHECK: zext i8
    e as isize
}

// CHECK-LABEL: @discriminant_2
#[no_mangle]
pub fn discriminant_2(e: Enum2) -> isize {
    // CHECK: zext i8
    e as isize
}
