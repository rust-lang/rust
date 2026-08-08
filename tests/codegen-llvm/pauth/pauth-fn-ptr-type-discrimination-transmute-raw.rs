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
// Transmutes through raw pointers. Make sure we match clang's behavior of treating raw pointers as
// zero-discriminated.
//
// Equivalent C:
// #include <stdint.h>
//
// void f(void) {}
// void g(int x) {}
//
// typedef void (*fn_int)(int);
// typedef void (*fn_void)(void);
//
// void test_fn_ptr_raw_ptr_fn_ptr(void) {
//   fn_int p = g;
//
//   void *raw = (void *)(uintptr_t)p;
//
//   fn_int q = (fn_int)(uintptr_t)raw;
//
//   q(123);
// }
//
// // C has no notion of mutable pointer, but keep it for consistency.
// void test_round_trip_through_mutable_raw_ptr(void) {
//   fn_void p = f;
//
//   void *raw = (void *)(uintptr_t)p;
//
//   fn_void q = (fn_void)(uintptr_t)raw;
//
//   q();
// }

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

extern crate minicore;

use minicore::mem::transmute;
use minicore::ptr;

extern "C" fn f() {}
extern "C" fn g(_: i32) {}

// CHECK-LABEL-DAG: test_fn_ptr_raw_ptr_fn_ptr
pub fn test_fn_ptr_raw_ptr_fn_ptr() {
    unsafe {
        // DISC: call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712) to i64), i32 0, i64 2712, i32 0, i64 0)
        let p: extern "C" fn(i32) = g;
        let raw: *const () = transmute(p);

        // DISC: = call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 0, i32 0, i64 2712)
        let q: extern "C" fn(i32) = transmute(raw);

        // DISC: call void %{{.*}}(i32 123) #[[#]] [ "ptrauth"(i32 0, i64 2712) ]
        // NO_DISC: call void ptrauth (ptr @{{.*}}g, i32 0)(i32 123) #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        q(123);
    }
}

// CHECK-LABEL-DAG: test_round_trip_through_mutable_raw_ptr
pub fn test_round_trip_through_mutable_raw_ptr() {
    unsafe {
        // DISC: call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @{{.*}}f, i32 0, i64 18983) to i64), i32 0, i64 18983, i32 0, i64 0)
        let p: extern "C" fn() = f;
        let raw: *mut () = transmute(p);

        // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 0, i32 0, i64 18983)
        let q: extern "C" fn() = transmute(raw);

        // DISC: call void %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 18983) ]
        // NO_DISC: call void ptrauth (ptr @{{.*}}f, i32 0)() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        q();
    }
}
