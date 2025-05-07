// Checks that range metadata gets emitted on functions result and arguments
// with scalar value.

// 32-bit systems will return 128bit values using a return area pointer.
//@ revisions: bit32 bit64
//@[bit32] only-32bit
//@[bit64] only-64bit
//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes

#![crate_type = "lib"]

use std::num::NonZero;

// Hack to get the correct size for usize
// CHECK: @helper([[USIZE:i[0-9]+]] noundef %_1)
#[no_mangle]
pub fn helper(_: usize) {}

// bit32: void @nonzero_int({{.*}} sret([16 x i8]) {{.*}}, i128 noundef range(i128 1, 0) %x)
// bit64: noundef range(i128 1, 0) i128 @nonzero_int(i128 noundef range(i128 1, 0) %x)
#[no_mangle]
pub fn nonzero_int(x: NonZero<u128>) -> NonZero<u128> {
    x
}

// CHECK: noundef range(i8 0, 3) i8 @optional_bool(i8{{.*}} range(i8 0, 3) %x)
#[no_mangle]
pub fn optional_bool(x: Option<bool>) -> Option<bool> {
    x
}

pub enum Enum0 {
    A(bool),
    B,
    C,
}

// CHECK: noundef range(i8 0, 4) i8 @enum0_value(i8{{.*}} range(i8 0, 4) %x)
#[no_mangle]
pub fn enum0_value(x: Enum0) -> Enum0 {
    x
}

pub enum Enum1 {
    A(u64),
    B(u64),
    C(u64),
}

// bit32: void @enum1_value({{.*}} sret({{[^,]*}}) {{[^,]*}}, [[ENUM1_TYP:i[0-9]+]]
// bit64: { [[ENUM1_TYP:i[0-9]+]], i64 } @enum1_value([[ENUM1_TYP]]
// CHECK-SAME: noundef range([[ENUM1_TYP]] 0, 3) %x.0, i64 noundef %x.1)
#[no_mangle]
pub fn enum1_value(x: Enum1) -> Enum1 {
    x
}

pub enum Enum2 {
    A(Enum0),
    B(Enum0),
    C(Enum0),
}

// CHECK: { i8, i8 } @enum2_value(i8 noundef range(i8 0, 3) %x.0, i8 noundef %x.1)
#[no_mangle]
pub fn enum2_value(x: Enum2) -> Enum2 {
    x
}

// CHECK: noundef [[USIZE]] @takes_slice(ptr noalias noundef nonnull readonly align 4 %x.0, [[USIZE]] noundef %x.1)
#[no_mangle]
pub fn takes_slice(x: &[i32]) -> usize {
    x.len()
}
