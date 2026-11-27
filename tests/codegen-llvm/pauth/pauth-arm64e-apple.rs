//@ add-minicore
//@ needs-llvm-components: aarch64
//@ revisions: IOS DARWIN TVOS
//@ compile-flags: -Copt-level=0 -Cpanic=abort
//@ [IOS] compile-flags: --target=arm64e-apple-ios
//@ [DARWIN] compile-flags: --target=arm64e-apple-darwin
//@ [TVOS] compile-flags: --target=arm64e-apple-tvos
//@ filecheck-flags: --implicit-check-not=aarch64-elf-pauthabi
//@ filecheck-flags: --implicit-check-not=ptrauth-elf-got
//@ filecheck-flags: --implicit-check-not=ptrauth-sign-personality

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;

extern "C" {
    fn external_c();
}

// CHECK: @C_POINTER = {{.*}}ptr ptrauth (ptr @external_c, i32 0)
#[no_mangle]
pub static C_POINTER: unsafe extern "C" fn() = external_c;

// CHECK-LABEL: define {{.*}} @call_c(
// CHECK-SAME: #[[ATTR:[0-9]+]]
// CHECK: call void %{{.*}}(){{.*}}[ "ptrauth"(i32 0, i64 0) ]
#[no_mangle]
pub unsafe extern "C" fn call_c(f: unsafe extern "C" fn()) {
    f();
}

// CHECK-LABEL: define {{.*}} @call_system(
// CHECK: call void %{{.*}}(){{.*}}[ "ptrauth"(i32 0, i64 0) ]
#[no_mangle]
pub unsafe fn call_system(f: unsafe extern "system" fn()) {
    f();
}

// CHECK-LABEL: define {{.*}} @take_c(
// CHECK: ret ptr ptrauth (ptr @external_c, i32 0)
#[no_mangle]
pub fn take_c() -> unsafe extern "C" fn() {
    external_c
}

// CHECK: attributes #[[ATTR]] = { {{.*}}"aarch64-jump-table-hardening"
// CHECK-SAME: "ptrauth-auth-traps"
// CHECK-SAME: "ptrauth-calls"
// CHECK-SAME: "ptrauth-indirect-gotos"
// CHECK-SAME: "ptrauth-returns"
