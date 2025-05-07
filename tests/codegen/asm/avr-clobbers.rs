//@ add-core-stubs
//@ assembly-output: emit-asm
//@ compile-flags: --target avr-none -C target-cpu=atmega328p
//@ needs-llvm-components: avr

#![crate_type = "rlib"]
#![feature(no_core, asm_experimental_arch)]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: @sreg_is_clobbered
// CHECK: void asm sideeffect "", "~{sreg}"()
#[no_mangle]
pub unsafe fn sreg_is_clobbered() {
    asm!("", options(nostack, nomem));
}

// CHECK-LABEL: @sreg_is_not_clobbered_if_preserve_flags_is_used
// CHECK: void asm sideeffect "", ""()
#[no_mangle]
pub unsafe fn sreg_is_not_clobbered_if_preserve_flags_is_used() {
    asm!("", options(nostack, nomem, preserves_flags));
}

// CHECK-LABEL: @clobber_abi
// CHECK: asm sideeffect "", "={r18},={r19},={r20},={r21},={r22},={r23},={r24},={r25},={r26},={r27},={r30},={r31},~{sreg}"()
#[no_mangle]
pub unsafe fn clobber_abi() {
    asm!("", clobber_abi("C"), options(nostack, nomem));
}

// CHECK-LABEL: @clobber_abi_with_preserved_flags
// CHECK: asm sideeffect "", "={r18},={r19},={r20},={r21},={r22},={r23},={r24},={r25},={r26},={r27},={r30},={r31}"()
#[no_mangle]
pub unsafe fn clobber_abi_with_preserved_flags() {
    asm!("", clobber_abi("C"), options(nostack, nomem, preserves_flags));
}
