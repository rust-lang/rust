//@ add-minicore
//@ revisions: thumbv8m_hf armv7_hf armv7_hf_d32
//
//@[thumbv8m_hf] compile-flags: --target thumbv8m.main-none-eabihf
//@[armv7_hf] compile-flags: --target armv7-unknown-linux-gnueabihf
//@[armv7_hf_d32] compile-flags: --target armv7-unknown-linux-gnueabihf -Ctarget-feature=+d32
//
//@ needs-llvm-components: arm
//
// ignore-tidy-linelength

#![crate_type = "rlib"]
#![feature(no_core, arm_target_feature)]
#![no_core]

extern crate minicore;
use minicore::*;

// Without the d32 target feature (implied by e.g. neon) the d16..=d31 registers are not available.
// They are marked as reserved in LLVM, and should not be clobbered.

// CHECK-LABEL: @clobber_abi
// CHECK: asm sideeffect alignstack "nop"
// CHECK: ={s0}
// CHECK: ={s15}
// thumbv8m_hf-NOT: ~{d16}
// armv7_hf-NOT: ~{d16}
// armv7_hf_d32: ={d16}
#[unsafe(no_mangle)]
pub unsafe fn clobber_abi() {
    asm!("nop", clobber_abi("C"));
}

// armv7_hf-LABEL: @clobber_abi_target_feature_d32
// armv7_hf: ={d16}
// armv7_hf_d32-LABEL: @clobber_abi_target_feature_d32
// armv7_hf_d32: ={d16}
#[cfg(not(thumbv8m_hf))]
#[target_feature(enable = "d32")]
#[unsafe(no_mangle)]
pub unsafe fn clobber_abi_target_feature_d32() {
    asm!("nop", clobber_abi("C"));
}

// CHECK-LABEL: @clobber_explicit_d16
// thumbv8m_hf-NOT: {d16}
// armv7_hf-NOT: {d16}
// armv7_hf_d32: =&{d16}
#[unsafe(no_mangle)]
pub unsafe fn clobber_explicit_d16() {
    asm!("nop", out("d16") _);
}
