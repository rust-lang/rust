// ignore-tidy-file-linelength
//@ add-minicore
// Run it at O0, so that the compiler doesn't optimise the calls away.
//@ revisions: DISC NO_DISC

//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0

// A trait method may be `extern "C"`. The vtable entry of an `extern "C"` method is signed with the
// discriminator of the impl method, and a virtual call authenticates with the discriminator of the
// trait method. The two signatures only differ in the type of `self`, which is a pointer in both,
// so the discriminators are equal.
// Rust ABI methods stay unsigned.
//
// Discriminators: `extern "C" fn(&S) -> i32` = 12410.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

pub trait T {
    fn rust_method(&self) -> i32;
    extern "C" fn c_method(&self) -> i32;
}

pub struct S;

impl T for S {
    fn rust_method(&self) -> i32 {
        1
    }
    extern "C" fn c_method(&self) -> i32 {
        2
    }
}

// DISC: @vtable.{{[0-9]+}} = {{.*}}ptr @{{.*}}rust_method{{.*}}, ptr ptrauth (ptr @{{.*}}c_method{{.*}}, i32 0, i64 12410)
// NO_DISC: @vtable.{{[0-9]+}} = {{.*}}ptr @{{.*}}rust_method{{.*}}, ptr ptrauth (ptr @{{.*}}c_method{{.*}}, i32 0)

// CHECK-LABEL: @make(
#[no_mangle]
pub fn make() -> &'static dyn T {
    &S
}

// CHECK-LABEL: @call_c_method(
#[no_mangle]
pub fn call_c_method(x: &dyn T) -> i32 {
    // DISC: call i32 %{{.*}}(ptr {{.*}}) {{.*}}[ "ptrauth"(i32 0, i64 12410) ]
    // NO_DISC: call i32 %{{.*}}(ptr {{.*}}) {{.*}}[ "ptrauth"(i32 0, i64 0) ]
    x.c_method()
}

// CHECK-LABEL: @call_rust_method(
#[no_mangle]
pub fn call_rust_method(x: &dyn T) -> i32 {
    // CHECK-NOT: "ptrauth"
    // CHECK: ret i32
    x.rust_method()
}
