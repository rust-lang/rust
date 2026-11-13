//@ add-minicore
//@ assembly-output: emit-asm
//@ needs-llvm-components: aarch64
//@ revisions: IOS DARWIN
//@ [IOS] compile-flags: --target=arm64e-apple-ios
//@ [DARWIN] compile-flags: --target=arm64e-apple-darwin
//@ compile-flags: -Copt-level=0 -Cpanic=abort

#![feature(no_core, thread_local)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

extern "C" {
    fn external_target(value: usize) -> usize;
    #[thread_local]
    static EXTERNAL_TLS: usize;
}

// CHECK-LABEL: _call_direct:
// CHECK: pacibsp
// CHECK: bl _external_target
// CHECK: retab
#[no_mangle]
pub unsafe extern "C" fn call_direct(value: usize) -> usize {
    external_target(value)
}

// CHECK-LABEL: _call_indirect:
// CHECK: pacibsp
// CHECK: blraaz x{{[0-9]+}}
// CHECK: retab
#[no_mangle]
pub extern "C" fn call_indirect(f: extern "C" fn(usize) -> usize, value: usize) -> usize {
    f(value)
}

// CHECK-LABEL: _load_tls:
// CHECK: pacibsp
// CHECK: blraaz x{{[0-9]+}}
// CHECK: retab
#[no_mangle]
pub unsafe extern "C" fn load_tls() -> usize {
    EXTERNAL_TLS
}

// CHECK-LABEL: _SIGNED_POINTER:
// CHECK: .quad _external_target@AUTH(ia,0)
#[no_mangle]
pub static SIGNED_POINTER: unsafe extern "C" fn(usize) -> usize = external_target;
