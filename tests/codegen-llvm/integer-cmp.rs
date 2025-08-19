// This is test for more optimal Ord implementation for integers.
// See <https://github.com/rust-lang/rust/issues/63758> for more info.

//@ revisions: llvm-pre-20 llvm-20
//@ [llvm-20] min-llvm-version: 20
//@ [llvm-pre-20] max-llvm-major-version: 19
//@ compile-flags: -C opt-level=3 -Zmerge-functions=disabled

#![crate_type = "lib"]

use std::cmp::Ordering;

// CHECK-LABEL: @cmp_signed
#[no_mangle]
pub fn cmp_signed(a: i64, b: i64) -> Ordering {
    // llvm-20: call{{.*}} i8 @llvm.scmp.i8.i64
    // llvm-pre-20: icmp slt
    // llvm-pre-20: icmp ne
    // llvm-pre-20: zext i1
    // llvm-pre-20: select i1
    a.cmp(&b)
}

// CHECK-LABEL: @cmp_unsigned
#[no_mangle]
pub fn cmp_unsigned(a: u32, b: u32) -> Ordering {
    // llvm-20: call{{.*}} i8 @llvm.ucmp.i8.i32
    // llvm-pre-20: icmp ult
    // llvm-pre-20: icmp ne
    // llvm-pre-20: zext i1
    // llvm-pre-20: select i1
    a.cmp(&b)
}

// CHECK-LABEL: @cmp_char
#[no_mangle]
pub fn cmp_char(a: char, b: char) -> Ordering {
    // llvm-20: call{{.*}} i8 @llvm.ucmp.i8.i32
    // llvm-pre-20: icmp ult
    // llvm-pre-20: icmp ne
    // llvm-pre-20: zext i1
    // llvm-pre-20: select i1
    a.cmp(&b)
}

// CHECK-LABEL: @cmp_tuple
#[no_mangle]
pub fn cmp_tuple(a: (i16, u16), b: (i16, u16)) -> Ordering {
    // llvm-20-DAG: call{{.*}} i8 @llvm.ucmp.i8.i16
    // llvm-20-DAG: call{{.*}} i8 @llvm.scmp.i8.i16
    // llvm-20: ret i8
    // llvm-pre-20: icmp slt
    // llvm-pre-20: icmp ne
    // llvm-pre-20: zext i1
    // llvm-pre-20: select i1
    // llvm-pre-20: icmp ult
    // llvm-pre-20: icmp ne
    // llvm-pre-20: zext i1
    // llvm-pre-20: select i1
    // llvm-pre-20: select i1
    a.cmp(&b)
}
