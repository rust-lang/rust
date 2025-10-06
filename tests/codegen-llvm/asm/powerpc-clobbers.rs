//@ add-core-stubs
//@ revisions: powerpc powerpc64 powerpc64le aix64
//@[powerpc] compile-flags: --target powerpc-unknown-linux-gnu
//@[powerpc] needs-llvm-components: powerpc
//@[powerpc64] compile-flags: --target powerpc64-unknown-linux-gnu
//@[powerpc64] needs-llvm-components: powerpc
//@[powerpc64le] compile-flags: --target powerpc64le-unknown-linux-gnu
//@[powerpc64le] needs-llvm-components: powerpc
//@[aix64] compile-flags: --target powerpc64-ibm-aix
//@[aix64] needs-llvm-components: powerpc
// ignore-tidy-linelength

#![crate_type = "rlib"]
#![feature(no_core, asm_experimental_arch)]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: @cr_clobber
// CHECK: call void asm sideeffect "", "~{cr}"()
#[no_mangle]
pub unsafe fn cr_clobber() {
    asm!("", out("cr") _, options(nostack, nomem, preserves_flags));
}

// CHECK-LABEL: @cr0_clobber
// CHECK: call void asm sideeffect "", "~{cr0}"()
#[no_mangle]
pub unsafe fn cr0_clobber() {
    asm!("", out("cr0") _, options(nostack, nomem, preserves_flags));
}

// CHECK-LABEL: @cr5_clobber
// CHECK: call void asm sideeffect "", "~{cr5}"()
#[no_mangle]
pub unsafe fn cr5_clobber() {
    asm!("", out("cr5") _, options(nostack, nomem, preserves_flags));
}

// CHECK-LABEL: @xer_clobber
// CHECK: call void asm sideeffect "", "~{xer}"()
#[no_mangle]
pub unsafe fn xer_clobber() {
    asm!("", out("xer") _, options(nostack, nomem, preserves_flags));
}

// Output format depends on the availability of altivec.
// CHECK-LABEL: @v0_clobber
// powerpc: call void asm sideeffect "", "~{v0}"()
// powerpc64: call <4 x i32> asm sideeffect "", "=&{v0}"()
// powerpc64le: call <4 x i32> asm sideeffect "", "=&{v0}"()
// aix64: call <4 x i32> asm sideeffect "", "=&{v0}"()
#[no_mangle]
pub unsafe fn v0_clobber() {
    asm!("", out("v0") _, options(nostack, nomem, preserves_flags));
}

// Output format depends on the availability of altivec.
// CHECK-LABEL: @clobber_abi
// powerpc: asm sideeffect "", "={r0},={r3},={r4},={r5},={r6},={r7},={r8},={r9},={r10},={r11},={r12},={f0},={f1},={f2},={f3},={f4},={f5},={f6},={f7},={f8},={f9},={f10},={f11},={f12},={f13},~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{cr0},~{cr1},~{cr5},~{cr6},~{cr7},~{ctr},~{lr},~{xer}"()
// powerpc64: asm sideeffect "", "={r0},={r3},={r4},={r5},={r6},={r7},={r8},={r9},={r10},={r11},={r12},={f0},={f1},={f2},={f3},={f4},={f5},={f6},={f7},={f8},={f9},={f10},={f11},={f12},={f13},={v0},={v1},={v2},={v3},={v4},={v5},={v6},={v7},={v8},={v9},={v10},={v11},={v12},={v13},={v14},={v15},={v16},={v17},={v18},={v19},~{cr0},~{cr1},~{cr5},~{cr6},~{cr7},~{ctr},~{lr},~{xer}"()
// powerpc64le: asm sideeffect "", "={r0},={r3},={r4},={r5},={r6},={r7},={r8},={r9},={r10},={r11},={r12},={f0},={f1},={f2},={f3},={f4},={f5},={f6},={f7},={f8},={f9},={f10},={f11},={f12},={f13},={v0},={v1},={v2},={v3},={v4},={v5},={v6},={v7},={v8},={v9},={v10},={v11},={v12},={v13},={v14},={v15},={v16},={v17},={v18},={v19},~{cr0},~{cr1},~{cr5},~{cr6},~{cr7},~{ctr},~{lr},~{xer}"()
// aix64: asm sideeffect "", "={r0},={r3},={r4},={r5},={r6},={r7},={r8},={r9},={r10},={r11},={r12},={f0},={f1},={f2},={f3},={f4},={f5},={f6},={f7},={f8},={f9},={f10},={f11},={f12},={f13},={v0},={v1},={v2},={v3},={v4},={v5},={v6},={v7},={v8},={v9},={v10},={v11},={v12},={v13},={v14},={v15},={v16},={v17},={v18},={v19},~{cr0},~{cr1},~{cr5},~{cr6},~{cr7},~{ctr},~{lr},~{xer}"()
#[no_mangle]
pub unsafe fn clobber_abi() {
    asm!("", clobber_abi("C"), options(nostack, nomem, preserves_flags));
}
