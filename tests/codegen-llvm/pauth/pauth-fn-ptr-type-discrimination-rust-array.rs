// ignore-tidy-file-linelength
//@ add-minicore
//@ only-pauthtest
// Run it at O0, so that the compiler doesn't optimise the calls away.

//@ revisions: DISC NO_DISC
//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0

// Make sure that signing/auth happens for every element of an array.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::mem;

type Fn = extern "C" fn(i32) -> i32;

#[repr(C)]
pub struct S {
    pub f: Fn,
    pub x: u32,
}

extern "C" fn foo(x: i32) -> i32 {
    x + 1
}

#[used]
// DISC: @{{.*}}TEST_ARR = {{.*}} ptr ptrauth (ptr @{{.*}}foo, i32 0, i64 2981), {{.*}} ptr ptrauth (ptr @{{.*}}foo, i32 0, i64 2981), {{.*}} ptr ptrauth (ptr @{{.*}}foo, i32 0, i64 2981)
// NO_DISC: @{{.*}}TEST_ARR = {{.*}} ptr ptrauth (ptr @{{.*}}foo, i32 0), {{.*}} ptr ptrauth (ptr @{{.*}}foo, i32 0), {{.*}} ptr ptrauth (ptr @{{.*}}foo, i32 0)
static TEST_ARR: [S; 3] = [S { f: foo, x: 1 }, S { f: foo, x: 2 }, S { f: foo, x: 3 }];

#[inline(never)]
// CHECK-LABEL: use_array
pub fn use_array(arr: &[S; 3]) -> i32 {
    let [a, b, c] = arr;
    // DISC: call i32 {{.*}} [ "ptrauth"(i32 0, i64 2981) ]
    // DISC: call i32 {{.*}} [ "ptrauth"(i32 0, i64 2981) ]
    // DISC: call i32 {{.*}} [ "ptrauth"(i32 0, i64 2981) ]
    // NO_DISC: call i32 {{.*}} [ "ptrauth"(i32 0, i64 0) ]
    // NO_DISC: call i32 {{.*}} [ "ptrauth"(i32 0, i64 0) ]
    // NO_DISC: call i32 {{.*}} [ "ptrauth"(i32 0, i64 0) ]
    (a.f)(a.x as i32) + (b.f)(b.x as i32) + (c.f)(c.x as i32)
}

#[no_mangle]
// CHECK-LABEL: test
pub fn test() -> i32 {
    // DISC: store ptr ptrauth (ptr @{{.*}}foo, i32 0, i64 2981)
    // NO_DISC: store ptr ptrauth (ptr @{{.*}}foo, i32 0)
    // CHECK: store i32 1
    // DISC: store ptr ptrauth (ptr @{{.*}}foo, i32 0, i64 2981)
    // NO_DISC: store ptr ptrauth (ptr @{{.*}}foo, i32 0)
    // CHECK: store i32 2
    // DISC: store ptr ptrauth (ptr @{{.*}}foo, i32 0, i64 2981)
    // NO_DISC: store ptr ptrauth (ptr @{{.*}}foo, i32 0)
    // CHECK: store i32 3
    let TEST_LOCAL_ARR: [S; 3] = [S { f: foo, x: 1 }, S { f: foo, x: 2 }, S { f: foo, x: 3 }];
    use_array(&TEST_ARR) + use_array(&TEST_LOCAL_ARR)
}
