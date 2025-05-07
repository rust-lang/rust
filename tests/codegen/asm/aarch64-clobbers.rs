//@ add-core-stubs
//@ revisions: aarch64 aarch64_fixed_x18 aarch64_no_x18 aarch64_reserve_x18 arm64ec
//@[aarch64] compile-flags: --target aarch64-unknown-linux-gnu
//@[aarch64] needs-llvm-components: aarch64
//@[aarch64_fixed_x18] compile-flags: --target aarch64-unknown-linux-gnu -Zfixed-x18
//@[aarch64_fixed_x18] needs-llvm-components: aarch64
//@[aarch64_no_x18] compile-flags: --target aarch64-pc-windows-msvc
//@[aarch64_no_x18] needs-llvm-components: aarch64
// aarch64-unknown-trusty uses aarch64-unknown-unknown-musl which doesn't
// reserve x18 by default as llvm_target, and pass +reserve-x18 in target-spec.
//@[aarch64_reserve_x18] compile-flags: --target aarch64-unknown-trusty
//@[aarch64_reserve_x18] needs-llvm-components: aarch64
//@[arm64ec] compile-flags: --target arm64ec-pc-windows-msvc
//@[arm64ec] needs-llvm-components: aarch64
// ignore-tidy-linelength

#![crate_type = "rlib"]
#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: @cc_clobber
// CHECK: call void asm sideeffect "", "~{cc}"()
#[no_mangle]
pub unsafe fn cc_clobber() {
    asm!("", options(nostack, nomem));
}

// CHECK-LABEL: @no_clobber
// CHECK: call void asm sideeffect "", ""()
#[no_mangle]
pub unsafe fn no_clobber() {
    asm!("", options(nostack, nomem, preserves_flags));
}

// CHECK-LABEL: @clobber_abi
// aarch64: asm sideeffect "", "={w0},={w1},={w2},={w3},={w4},={w5},={w6},={w7},={w8},={w9},={w10},={w11},={w12},={w13},={w14},={w15},={w16},={w17},={w18},={w30},={q0},={q1},={q2},={q3},={q4},={q5},={q6},={q7},={q8},={q9},={q10},={q11},={q12},={q13},={q14},={q15},={q16},={q17},={q18},={q19},={q20},={q21},={q22},={q23},={q24},={q25},={q26},={q27},={q28},={q29},={q30},={q31},~{p0},~{p1},~{p2},~{p3},~{p4},~{p5},~{p6},~{p7},~{p8},~{p9},~{p10},~{p11},~{p12},~{p13},~{p14},~{p15},~{ffr}"()
// aarch64_fixed_x18: asm sideeffect "", "={w0},={w1},={w2},={w3},={w4},={w5},={w6},={w7},={w8},={w9},={w10},={w11},={w12},={w13},={w14},={w15},={w16},={w17},={w30},={q0},={q1},={q2},={q3},={q4},={q5},={q6},={q7},={q8},={q9},={q10},={q11},={q12},={q13},={q14},={q15},={q16},={q17},={q18},={q19},={q20},={q21},={q22},={q23},={q24},={q25},={q26},={q27},={q28},={q29},={q30},={q31},~{p0},~{p1},~{p2},~{p3},~{p4},~{p5},~{p6},~{p7},~{p8},~{p9},~{p10},~{p11},~{p12},~{p13},~{p14},~{p15},~{ffr}"()
// aarch64_no_x18: asm sideeffect "", "={w0},={w1},={w2},={w3},={w4},={w5},={w6},={w7},={w8},={w9},={w10},={w11},={w12},={w13},={w14},={w15},={w16},={w17},={w30},={q0},={q1},={q2},={q3},={q4},={q5},={q6},={q7},={q8},={q9},={q10},={q11},={q12},={q13},={q14},={q15},={q16},={q17},={q18},={q19},={q20},={q21},={q22},={q23},={q24},={q25},={q26},={q27},={q28},={q29},={q30},={q31},~{p0},~{p1},~{p2},~{p3},~{p4},~{p5},~{p6},~{p7},~{p8},~{p9},~{p10},~{p11},~{p12},~{p13},~{p14},~{p15},~{ffr}"()
// aarch64_reserve_x18: asm sideeffect "", "={w0},={w1},={w2},={w3},={w4},={w5},={w6},={w7},={w8},={w9},={w10},={w11},={w12},={w13},={w14},={w15},={w16},={w17},={w30},={q0},={q1},={q2},={q3},={q4},={q5},={q6},={q7},={q8},={q9},={q10},={q11},={q12},={q13},={q14},={q15},={q16},={q17},={q18},={q19},={q20},={q21},={q22},={q23},={q24},={q25},={q26},={q27},={q28},={q29},={q30},={q31},~{p0},~{p1},~{p2},~{p3},~{p4},~{p5},~{p6},~{p7},~{p8},~{p9},~{p10},~{p11},~{p12},~{p13},~{p14},~{p15},~{ffr}"()
// arm64ec: asm sideeffect "", "={w0},={w1},={w2},={w3},={w4},={w5},={w6},={w7},={w8},={w9},={w10},={w11},={w12},={w15},={w16},={w17},={w30},={q0},={q1},={q2},={q3},={q4},={q5},={q6},={q7},={q8},={q9},={q10},={q11},={q12},={q13},={q14},={q15}"()
#[no_mangle]
pub unsafe fn clobber_abi() {
    asm!("", clobber_abi("C"), options(nostack, nomem, preserves_flags));
}
