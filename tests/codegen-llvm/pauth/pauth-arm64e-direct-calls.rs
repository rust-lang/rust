//@ add-minicore
//@ needs-llvm-components: aarch64
//@ revisions: O0 O3
//@ compile-flags: --target=arm64e-apple-ios -Cpanic=abort
//@ [O0] compile-flags: -Copt-level=0
//@ [O3] compile-flags: -Copt-level=3

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;

extern "C" {
    fn external_c();
}

extern "Rust" {
    fn external_rust();
}

// CHECK-LABEL: define {{.*}} @direct_c(
// CHECK: call void @external_c()
// CHECK: ret void
#[no_mangle]
pub unsafe fn direct_c() {
    external_c();
}

// CHECK-LABEL: define {{.*}} @direct_rust(
// CHECK: call void @external_rust()
// CHECK: ret void
#[no_mangle]
pub unsafe fn direct_rust() {
    external_rust();
}

// CHECK-LABEL: define {{.*}} @take_c(
// CHECK: ret ptr ptrauth (ptr @external_c, i32 0)
#[no_mangle]
pub fn take_c() -> unsafe extern "C" fn() {
    external_c
}

// CHECK-LABEL: define {{.*}} @take_rust(
// CHECK: ret ptr ptrauth (ptr @external_rust, i32 0)
#[no_mangle]
pub fn take_rust() -> unsafe fn() {
    external_rust
}
