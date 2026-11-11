//@ add-minicore
//@ needs-llvm-components: aarch64
//@ revisions: ARM64 ARM64EC
//@ compile-flags: -Copt-level=0 -Cpanic=abort
//@ [ARM64] compile-flags: --target=aarch64-apple-darwin
//@ [ARM64EC] compile-flags: --target=arm64ec-pc-windows-msvc
//@ filecheck-flags: --implicit-check-not=ptrauth

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;

extern "C" {
    fn external_c();
}

// CHECK: @C_POINTER = {{.*}}ptr @external_c
#[no_mangle]
pub static C_POINTER: unsafe extern "C" fn() = external_c;

// CHECK-LABEL: define {{.*}} @call_c(
// CHECK: call void %{{.*}}()
#[no_mangle]
pub unsafe extern "C" fn call_c(f: unsafe extern "C" fn()) {
    f();
}
