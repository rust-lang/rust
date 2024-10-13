//@ assembly-output: emit-asm
//@ compile-flags: --target arm64ec-pc-windows-msvc
//@ needs-llvm-components: aarch64

#![crate_type = "rlib"]
#![feature(no_core, rustc_attrs, lang_items, asm_experimental_arch)]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}

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
// CHECK: asm sideeffect "", "={w0},={w1},={w2},={w3},={w4},={w5},={w6},={w7},={w8},={w9},={w10},={w11},={w12},={w15},={w16},={w17},={w30},={q0},={q1},={q2},={q3},={q4},={q5},={q6},={q7},={q8},={q9},={q10},={q11},={q12},={q13},={q14},={q15}"()
#[no_mangle]
pub unsafe fn clobber_abi() {
    asm!("", clobber_abi("C"), options(nostack, nomem, preserves_flags));
}
