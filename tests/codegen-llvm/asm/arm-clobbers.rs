//@ add-minicore
//@ revisions: vfp softfloat neon
//@[vfp] compile-flags: --target thumbv7em-none-eabihf
//@[vfp] needs-llvm-components: arm
//@[softfloat] compile-flags: --target thumbv7em-none-eabi
//@[softfloat] needs-llvm-components: arm
//@[neon] compile-flags: --target armv7-unknown-linux-gnueabihf -C target-feature=+neon
//@[neon] needs-llvm-components: arm

// Discarded `qN` outputs must clobber the overlapping `d` registers when the
// `qN` register itself is unavailable (no NEON), since LLVM silently ignores
// clobbers of registers whose class has no legal type.
// See <https://github.com/rust-lang/rust/issues/166311>.

#![crate_type = "rlib"]
#![feature(no_core, arm_target_feature)]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: @q4_clobber
// vfp: call void asm sideeffect "", "~{d8},~{d9}"()
// softfloat: call void asm sideeffect "", "~{q4}"()
// neon: call <2 x i64> asm sideeffect "", "=&{q4}"()
#[no_mangle]
pub unsafe fn q4_clobber() {
    asm!("", out("q4") _, options(nostack, nomem, preserves_flags));
}

// d16-d31 only exist with the `d32` feature.
// CHECK-LABEL: @q8_clobber
// vfp: call void asm sideeffect "", "~{q8}"()
// softfloat: call void asm sideeffect "", "~{q8}"()
// neon: call <2 x i64> asm sideeffect "", "=&{q8}"()
#[no_mangle]
pub unsafe fn q8_clobber() {
    asm!("", out("q8") _, options(nostack, nomem, preserves_flags));
}

// CHECK-LABEL: @q8_clobber_d32
// vfp: call void asm sideeffect "", "~{d16},~{d17}"()
// softfloat: call void asm sideeffect "", "~{d16},~{d17}"()
// neon: call <2 x i64> asm sideeffect "", "=&{q8}"()
#[no_mangle]
#[target_feature(enable = "d32")]
pub unsafe fn q8_clobber_d32() {
    asm!("", out("q8") _, options(nostack, nomem, preserves_flags));
}

// CHECK-LABEL: @q_clobber_multiple
// vfp: call void asm sideeffect "", "~{d8},~{d9},~{d10},~{d11},~{d12},~{d13},~{d14},~{d15}"()
// softfloat: call void asm sideeffect "", "~{q4},~{q5},~{q6},~{q7}"()
// neon: call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } asm sideeffect "", "=&{q4},=&{q5},=&{q6},=&{q7}"()
#[no_mangle]
pub unsafe fn q_clobber_multiple() {
    asm!(
        "",
        out("q4") _,
        out("q5") _,
        out("q6") _,
        out("q7") _,
        options(nostack, nomem, preserves_flags)
    );
}
