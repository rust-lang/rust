// ignore-tidy-file-linelength
//@ add-minicore
//@ only-pauthtest
// Run it at O0, so that the compiler doesn't optimise the calls away.

//@ revisions: DISC NO_DISC
//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0

// Test generation of function-pointer type discriminators. The discriminator values were obtained
// from Clang by compiling equivalent C code (included). Both compilers must generate identical
// values.
//
// Make sure that signing/auth happens for every element of an array.
//
// Equivalent C sample:
//
// ```c
// #include <stdint.h>
//
// typedef int32_t (*Fn)(int32_t);
//
// struct S {
//   Fn f;
//   uint32_t x;
// };
//
// int32_t foo(int32_t x) { return x + 1; }
//
// __attribute__((used)) static const struct S TEST_ARR[3] = {
//     {.f = foo, .x = 1},
//     {.f = foo, .x = 2},
//     {.f = foo, .x = 3},
// };
//
// __attribute__((noinline)) int32_t use_array(const struct S (*arr)[3]) {
//   const struct S *a = &(*arr)[0];
//   const struct S *b = &(*arr)[1];
//   const struct S *c = &(*arr)[2];
//
//   return a->f((int32_t)a->x) + b->f((int32_t)b->x) + c->f((int32_t)c->x);
// }
//
// int32_t test(void) {
//   struct S TEST_LOCAL_ARR[3];
//   TEST_LOCAL_ARR[0].f = foo;
//   TEST_LOCAL_ARR[0].x = 1;
//   TEST_LOCAL_ARR[1].f = foo;
//   TEST_LOCAL_ARR[1].x = 2;
//   TEST_LOCAL_ARR[2].f = foo;
//   TEST_LOCAL_ARR[2].x = 3;
//
//   return use_array(&TEST_ARR) + use_array(&TEST_LOCAL_ARR);
// }
// ```

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
