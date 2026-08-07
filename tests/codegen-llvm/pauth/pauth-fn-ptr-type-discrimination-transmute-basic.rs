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
// Transmute on function pointers only.
//
// Equivalent C:
// #include <stdint.h>
//
// int f_i32(int x) { return x; }
//
// void f_void(void) {}
//
// int (*returns_fp(void))(int) { return f_i32; }
//
// void takes_fp(void (*f)(void)) { f(); }
//
// void test_call_after_transmute(void) {
//   int (*p)(int) = f_i32;
//   float (*q)(float) = (float (*)(float))p;
//   q(1.0f);
// }
//
// void test_double_transmute(void) {
//   void (*a)(void) = f_void;
//   void (*b)(int) = (void (*)(int))a;
//   void (*c)(void) = (void (*)(void))b;
//   c();
// }
//
// void test_returned_fp(void) {
//   int (*p)(int) = returns_fp();
//   float (*q)(float) = (float (*)(float))p;
//   q(2.0f);
// }
//
// void test_argument_transmute(void) {
//   void (*p)(void) = (void (*)(void))f_i32;
//   takes_fp(p);
// }
//
// void test_identity_transmute(void) {
//   int (*p)(int) = f_i32;
//   int (*q)(int) = (int (*)(int))p;
//   q(123);
// }
//
// void test_mutable_reassignment(void) {
//   int (*p)(int) = f_i32;
//   void (*q)(void) = (void (*)(void))p;
//   p = (int (*)(int))q;
//   p(456);
// }

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

extern crate minicore;

use minicore::hint::black_box;
use minicore::mem::transmute;

// NO_DISC-NOT: llvm.ptrauth.resign

extern "C" fn f_i32(x: i32) -> i32 {
    x
}

extern "C" fn f_void() {}

// CHECK-LABEL-DAG: returns_fp
extern "C" fn returns_fp() -> extern "C" fn(i32) -> i32 {
    // DISC: ret ptr ptrauth (ptr @{{.*}}f_i32, i32 0, i64 2981)
    // NO_DISC: ret ptr ptrauth (ptr @{{.*}}f_i32, i32 0)
    f_i32
}

// CHECK-LABEL-DAG: takes_fp
extern "C" fn takes_fp(f: extern "C" fn()) {
    // DISC: call void %f() #[[#]] [ "ptrauth"(i32 0, i64 18983) ]
    // NO_DISC: call void %f() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
    f();
}

// CHECK-LABEL-DAG: @test_call_after_transmute
pub fn test_call_after_transmute() {
    unsafe {
        let p: extern "C" fn(i32) -> i32 = f_i32;

        // DISC: [[TRANSMUTED:%.*]] = call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @{{.*}}f_i32, i32 0, i64 2981) to i64), i32 0, i64 2981, i32 0, i64 28450)
        // DISC: [[INTTOPTR:%.*]] = inttoptr i64 [[TRANSMUTED]] to ptr
        let q: extern "C" fn(f32) -> f32 = transmute(p);
        // DISC: call float [[INTTOPTR]](float 1.000000e+00) #[[#]] [ "ptrauth"(i32 0, i64 28450) ]

        // NO_DISC: call float ptrauth (ptr @{{.*}}f_i32, i32 0)(float 1.000000e+00) #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        q(1.0);
    }
}

// CHECK-LABEL-DAG: @test_double_transmute
pub fn test_double_transmute() {
    unsafe {
        // DISC: [[TRANSMUTED_1:%.*]] = call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @{{.*}}f_void, i32 0, i64 18983) to i64), i32 0, i64 18983, i32 0, i64 2712)
        let a: extern "C" fn() = f_void;
        let b: extern "C" fn(i32) = transmute(a);
        // DISC: [[INTTOPTR_1:%.*]] = inttoptr i64 [[TRANSMUTED_1]] to ptr
        // DISC: [[PTRTOINT:%.*]] = ptrtoint ptr [[INTTOPTR_1]] to i64
        // DISC: [[TRANSMUTED_2:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[PTRTOINT]], i32 0, i64 2712, i32 0, i64 18983)
        let c: extern "C" fn() = transmute(b);
        // DISC: [[INTTOPTR_2:%.*]] = inttoptr i64 [[TRANSMUTED_2]] to ptr
        // DISC: call void [[INTTOPTR_2]]() #[[#]] [ "ptrauth"(i32 0, i64 18983) ]
        // NO_DISC: call void ptrauth (ptr @{{.*}}f_void, i32 0)() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        c();
    }
}

// CHECK-LABEL-DAG: @test_returned_fp
pub fn test_returned_fp() {
    unsafe {
        let p = returns_fp();
        // 34128 here is the discriminator for returns_fp. The function pointer that returns_fp
        // returns is discriminated with 2981 - that value should be used as input discriminator
        // for resign below.
        // DISC: [[P:%.*]] = call ptr ptrauth (ptr @{{.*}}returns_fp, i32 0, i64 34128)() #[[#]] [ "ptrauth"(i32 0, i64 34128) ]
        // DISC: [[INTTOPTR_1:%.*]] = ptrtoint ptr [[P]] to i64
        // DISC: [[TRANSMUTED:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[INTTOPTR_1]], i32 0, i64 2981, i32 0, i64 28450)
        let q: extern "C" fn(f32) -> f32 = transmute(p);
        // DISC: [[INTTOPTR_2:%.*]] = inttoptr i64 [[TRANSMUTED]] to ptr
        // DISC: call float [[INTTOPTR_2]](float 2.000000e+00) #[[#]] [ "ptrauth"(i32 0, i64 28450) ]
        // NO_DISC: [[PTR:%.*]] = call ptr ptrauth (ptr @{{.*}}returns_fp, i32 0)() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        // NO_DISC: call float [[PTR]](float 2.000000e+00) #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        q(2.0);
    }
}

// CHECK-LABEL-DAG: @test_argument_transmute
pub fn test_argument_transmute() {
    unsafe {
        // DISC: [[TRANSMUTED:%.*]] = call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @{{.*}}f_i32, i32 0, i64 2981) to i64), i32 0, i64 2981, i32 0, i64 18983)
        let p: extern "C" fn() = transmute(f_i32 as extern "C" fn(i32) -> i32);
        // DISC: [[P:%.*]] = inttoptr i64 [[TRANSMUTED]] to ptr
        // DISC: call void ptrauth (ptr @{{.*}}takes_fp, i32 0, i64 10942)(ptr [[P]]) #[[#]] [ "ptrauth"(i32 0, i64 10942) ]
        // NO_DISC: call void ptrauth (ptr @{{.*}}takes_fp, i32 0)(ptr ptrauth (ptr @{{.*}}f_i32, i32 0)) #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        takes_fp(p);
    }
}

// CHECK-LABEL-DAG: @test_identity_transmute
pub fn test_identity_transmute() {
    unsafe {
        let p: extern "C" fn(i32) -> i32 = f_i32;
        let q: extern "C" fn(i32) -> i32 = transmute(p);
        // Expect transmutes to be optimised out.
        // DISC: call i32 ptrauth (ptr @{{.*}}f_i32, i32 0, i64 2981)(i32 123) #[[#]] [ "ptrauth"(i32 0, i64 2981) ]
        // NO_DISC: call i32 ptrauth (ptr @{{.*}}f_i32, i32 0)(i32 123) #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        q(123);
    }
}

// CHECK-LABEL-DAG: @test_mutable_reassignment
pub fn test_mutable_reassignment() {
    unsafe {
        // DISC: store ptr ptrauth (ptr @{{.*}}f_i32, i32 0, i64 2981)
        // NO_DISC: store ptr ptrauth (ptr @{{.*}}f_i32, i32 0), ptr %p
        let mut p: extern "C" fn(i32) -> i32 = f_i32;
        // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 2981, i32 0, i64 18983)
        let q: extern "C" fn() = transmute(p);
        // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 18983, i32 0, i64 2981)
        p = transmute(q);
        // DISC: call i32 %{{.*}}(i32 456) #[[#]] [ "ptrauth"(i32 0, i64 2981) ]
        // NO_DISC: call i32 %{{.*}}(i32 456) #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        p(456);
    }
}
