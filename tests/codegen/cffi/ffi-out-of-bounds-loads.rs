//@ add-core-stubs
//@ revisions: linux apple
//@ min-llvm-version: 19
//@ compile-flags: -Copt-level=0 -Cno-prepopulate-passes -Zlint-llvm-ir

//@[linux] compile-flags: --target x86_64-unknown-linux-gnu
//@[linux] needs-llvm-components: x86
//@[apple] compile-flags: --target x86_64-apple-darwin
//@[apple] needs-llvm-components: x86

// Regression test for #29988

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(C)]
struct S {
    f1: i32,
    f2: i32,
    f3: i32,
}

extern "C" {
    fn foo(s: S);
}

// CHECK-LABEL: @test
#[no_mangle]
pub fn test() {
    let s = S { f1: 1, f2: 2, f3: 3 };
    unsafe {
        // CHECK: [[ALLOCA:%.+]] = alloca [16 x i8], align 8
        // CHECK: [[LOAD:%.+]] = load { i64, i32 }, ptr [[ALLOCA]], align 8
        // CHECK: call void @foo({ i64, i32 } [[LOAD]])
        foo(s);
    }
}
