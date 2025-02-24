//@ add-core-stubs
//@ compile-flags: --target bpfel-unknown-none
//@ needs-llvm-components: bpf

#![crate_type = "rlib"]
#![feature(no_core, asm_experimental_arch)]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: @flags_clobber
// CHECK: call void asm sideeffect "", ""()
#[no_mangle]
pub unsafe fn flags_clobber() {
    asm!("", options(nostack, nomem));
}

// CHECK-LABEL: @no_clobber
// CHECK: call void asm sideeffect "", ""()
#[no_mangle]
pub unsafe fn no_clobber() {
    asm!("", options(nostack, nomem, preserves_flags));
}

// CHECK-LABEL: @clobber_abi
// CHECK: asm sideeffect "", "={r0},={r1},={r2},={r3},={r4},={r5}"()
#[no_mangle]
pub unsafe fn clobber_abi() {
    asm!("", clobber_abi("C"), options(nostack, nomem, preserves_flags));
}
