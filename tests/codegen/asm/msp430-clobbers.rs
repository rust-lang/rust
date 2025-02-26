//@ add-core-stubs
//@ assembly-output: emit-asm
//@ compile-flags: --target msp430-none-elf
//@ needs-llvm-components: msp430

#![crate_type = "rlib"]
#![feature(no_core, asm_experimental_arch)]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: @sr_clobber
// CHECK: call void asm sideeffect "", "~{sr}"()
#[no_mangle]
pub unsafe fn sr_clobber() {
    asm!("", options(nostack, nomem));
}

// CHECK-LABEL: @no_clobber
// CHECK: call void asm sideeffect "", ""()
#[no_mangle]
pub unsafe fn no_clobber() {
    asm!("", options(nostack, nomem, preserves_flags));
}

// CHECK-LABEL: @clobber_abi
// CHECK: asm sideeffect "", "={r11},={r12},={r13},={r14},={r15}"()
#[no_mangle]
pub unsafe fn clobber_abi() {
    asm!("", clobber_abi("C"), options(nostack, nomem, preserves_flags));
}
