//@ revisions: linux apple
//@ min-llvm-version: 19
//@ compile-flags: -Copt-level=0 -Cno-prepopulate-passes -Zlint-llvm-ir -Cllvm-args=-lint-abort-on-error

//@[linux] compile-flags: --target x86_64-unknown-linux-gnu
//@[linux] needs-llvm-components: x86
//@[apple] compile-flags: --target x86_64-apple-darwin
//@[apple] needs-llvm-components: x86

// Regression test for #29988

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_std]
#![no_core]

#[lang = "sized"]
trait Sized {}
#[lang = "freeze"]
trait Freeze {}
#[lang = "copy"]
trait Copy {}

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
    // CHECK-NEXT:  [[START:.*:]]
    // CHECK-NEXT:    [[S:%.*]] = alloca [12 x i8], align 4
    // CHECK-NEXT:    store i32 1, ptr [[S]], align 4
    // CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds i8, ptr [[S]], i64 4
    // CHECK-NEXT:    store i32 2, ptr [[TMP0]], align 4
    // CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i8, ptr [[S]], i64 8
    // CHECK-NEXT:    store i32 3, ptr [[TMP1]], align 4
    // CHECK-NEXT:    [[TMP2:%.*]] = load i64, ptr [[S]], align 4
    // CHECK-NEXT:    [[TMP3:%.*]] = insertvalue { i64, i32 } poison, i64 [[TMP2]], 0
    // CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i8, ptr [[S]], i64 8
    // CHECK-NEXT:    [[TMP5:%.*]] = load i32, ptr [[TMP4]], align 4
    // CHECK-NEXT:    [[TMP6:%.*]] = insertvalue { i64, i32 } [[TMP3]], i32 [[TMP5]], 1
    // CHECK-NEXT:    call void @foo({ i64, i32 } [[TMP6]]) #[[ATTR2:[0-9]+]]
    // CHECK-NEXT:    ret void
    let s = S { f1: 1, f2: 2, f3: 3 };
    unsafe {
        foo(s);
    }
}
