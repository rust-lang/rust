// ignore-tidy-file-linelength
//@ add-minicore
//@ needs-llvm-components: aarch64
// Run it at O0, so that the compiler doesn't optimise the calls away.
//@ compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -C opt-level=0

// Function pointers are signed, and calls through them authenticated, for the same set of ABIs:
// `extern "C"` and `extern "system"` (see `is_ptrauth_signed_abi`).
//
// Discriminators: `extern "C" fn(i32)` = 2712.

// FIXME: jchlanda - decide if `efiapi` should also participate in pointer authentication.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

extern "C" fn c_fn(_: i32) {}

extern "efiapi" fn efi_fn(_: i32) {}

// CHECK-LABEL: @reify_c(
#[no_mangle]
pub fn reify_c() -> extern "C" fn(i32) {
    // CHECK: ret ptr ptrauth (ptr @{{.*}}c_fn{{.*}}, i32 0)
    c_fn
}

// CHECK-LABEL: @reify_efiapi(
#[no_mangle]
pub fn reify_efiapi() -> extern "efiapi" fn(i32) {
    // CHECK-NOT: ptrauth
    // CHECK: ret ptr @{{.*}}efi_fn
    efi_fn
}

// CHECK-LABEL: @call_c(
#[no_mangle]
pub fn call_c(f: extern "C" fn(i32)) {
    // CHECK: call void %{{.*}}(i32 1) {{.*}}[ "ptrauth"(i32 0, i64 0) ]
    f(1)
}

// CHECK-LABEL: @call_efiapi(
#[no_mangle]
pub fn call_efiapi(f: extern "efiapi" fn(i32)) {
    // CHECK-NOT: "ptrauth"
    // CHECK: call void %{{.*}}(i32 1)
    // CHECK-NOT: "ptrauth"
    // CHECK: ret void
    f(1)
}
