// This is test for more optimal Ord implementation for integers.
// See <https://github.com/rust-lang/rust/issues/63758> for more info.

//@ compile-flags: -C opt-level=3 -Zmerge-functions=disabled

#![crate_type = "lib"]

use std::cmp::Ordering;

// CHECK-LABEL: @cmp_signed
#[no_mangle]
pub fn cmp_signed(a: i64, b: i64) -> Ordering {
    // CHECK: call{{.*}} i8 @llvm.scmp.i8.i64
    a.cmp(&b)
}

// CHECK-LABEL: @cmp_unsigned
#[no_mangle]
pub fn cmp_unsigned(a: u32, b: u32) -> Ordering {
    // CHECK: call{{.*}} i8 @llvm.ucmp.i8.i32
    a.cmp(&b)
}

// CHECK-LABEL: @cmp_char
#[no_mangle]
pub fn cmp_char(a: char, b: char) -> Ordering {
    // CHECK: call{{.*}} i8 @llvm.ucmp.i8.i32
    a.cmp(&b)
}

// CHECK-LABEL: @cmp_tuple
#[no_mangle]
pub fn cmp_tuple(a: (i16, u16), b: (i16, u16)) -> Ordering {
    // CHECK-DAG: call{{.*}} i8 @llvm.ucmp.i8.i16
    // CHECK-DAG: call{{.*}} i8 @llvm.scmp.i8.i16
    // CHECK: ret i8
    a.cmp(&b)
}
