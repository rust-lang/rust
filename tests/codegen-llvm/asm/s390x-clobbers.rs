//@ add-core-stubs
//@ revisions: s390x
//@[s390x] compile-flags: --target s390x-unknown-linux-gnu -C target-cpu=z10
//@[s390x] needs-llvm-components: systemz

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

// CHECK-LABEL: @a2_clobber
// CHECK: call void asm sideeffect "", "~{a2}"()
#[no_mangle]
pub unsafe fn a2_clobber() {
    asm!("", out("a2") _, options(nostack, nomem, preserves_flags));
}

// CHECK-LABEL: @v0_clobber
// CHECK: call void asm sideeffect "", "~{v0}"()
#[no_mangle]
pub unsafe fn v0_clobber() {
    asm!("", out("v0") _, options(nostack, nomem, preserves_flags));
}

// CHECK-LABEL: @clobber_abi
// CHECK: asm sideeffect "", "={r0},={r1},={r2},={r3},={r4},={r5},={r14},={f0},={f1},={f2},={f3},={f4},={f5},={f6},={f7},~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31},~{a2},~{a3},~{a4},~{a5},~{a6},~{a7},~{a8},~{a9},~{a10},~{a11},~{a12},~{a13},~{a14},~{a15}"()
#[no_mangle]
pub unsafe fn clobber_abi() {
    asm!("", clobber_abi("C"), options(nostack, nomem, preserves_flags));
}
