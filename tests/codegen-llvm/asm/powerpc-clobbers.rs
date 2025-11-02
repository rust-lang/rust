//@ add-minicore
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

// Output format depends on the availability of vsx.
// CHECK-LABEL: @vs32_clobber
// powerpc: call void asm sideeffect "", "~{vs32}"()
// powerpc64: call void asm sideeffect "", "~{vs32}"()
// powerpc64le: call <4 x i32> asm sideeffect "", "=&{vs32}"()
// aix64: call <4 x i32> asm sideeffect "", "=&{vs32}"()
#[no_mangle]
pub unsafe fn vs32_clobber() {
    asm!("", out("vs32") _, options(nostack, nomem, preserves_flags));
}

// Output format depends on the availability of altivec and vsx
// CHECK-LABEL: @clobber_abi
// powerpc: asm sideeffect "", "={r0},={r3},={r4},={r5},={r6},={r7},={r8},={r9},={r10},={r11},={r12},={f0},={f1},={f2},={f3},={f4},={f5},={f6},={f7},={f8},={f9},={f10},={f11},={f12},={f13},~{vs0},~{vs1},~{vs2},~{vs3},~{vs4},~{vs5},~{vs6},~{vs7},~{vs8},~{vs9},~{vs10},~{vs11},~{vs12},~{vs13},~{vs14},~{vs15},~{vs16},~{vs17},~{vs18},~{vs19},~{vs20},~{vs21},~{vs22},~{vs23},~{vs24},~{vs25},~{vs26},~{vs27},~{vs28},~{vs29},~{vs30},~{vs31},~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{cr0},~{cr1},~{cr5},~{cr6},~{cr7},~{ctr},~{lr},~{xer}"()
// powerpc64: asm sideeffect "", "={r0},={r3},={r4},={r5},={r6},={r7},={r8},={r9},={r10},={r11},={r12},={f0},={f1},={f2},={f3},={f4},={f5},={f6},={f7},={f8},={f9},={f10},={f11},={f12},={f13},={v0},={v1},={v2},={v3},={v4},={v5},={v6},={v7},={v8},={v9},={v10},={v11},={v12},={v13},={v14},={v15},={v16},={v17},={v18},={v19},~{vs0},~{vs1},~{vs2},~{vs3},~{vs4},~{vs5},~{vs6},~{vs7},~{vs8},~{vs9},~{vs10},~{vs11},~{vs12},~{vs13},~{vs14},~{vs15},~{vs16},~{vs17},~{vs18},~{vs19},~{vs20},~{vs21},~{vs22},~{vs23},~{vs24},~{vs25},~{vs26},~{vs27},~{vs28},~{vs29},~{vs30},~{vs31},~{cr0},~{cr1},~{cr5},~{cr6},~{cr7},~{ctr},~{lr},~{xer}"()
// powerpc64le: asm sideeffect "", "={r0},={r3},={r4},={r5},={r6},={r7},={r8},={r9},={r10},={r11},={r12},={f0},={f1},={f2},={f3},={f4},={f5},={f6},={f7},={f8},={f9},={f10},={f11},={f12},={f13},={vs0},={vs1},={vs2},={vs3},={vs4},={vs5},={vs6},={vs7},={vs8},={vs9},={vs10},={vs11},={vs12},={vs13},={vs14},={vs15},={vs16},={vs17},={vs18},={vs19},={vs20},={vs21},={vs22},={vs23},={vs24},={vs25},={vs26},={vs27},={vs28},={vs29},={vs30},={vs31},={v0},={v1},={v2},={v3},={v4},={v5},={v6},={v7},={v8},={v9},={v10},={v11},={v12},={v13},={v14},={v15},={v16},={v17},={v18},={v19},~{cr0},~{cr1},~{cr5},~{cr6},~{cr7},~{ctr},~{lr},~{xer}"()
// aix64: asm sideeffect "", "={r0},={r3},={r4},={r5},={r6},={r7},={r8},={r9},={r10},={r11},={r12},={f0},={f1},={f2},={f3},={f4},={f5},={f6},={f7},={f8},={f9},={f10},={f11},={f12},={f13},={vs0},={vs1},={vs2},={vs3},={vs4},={vs5},={vs6},={vs7},={vs8},={vs9},={vs10},={vs11},={vs12},={vs13},={vs14},={vs15},={vs16},={vs17},={vs18},={vs19},={vs20},={vs21},={vs22},={vs23},={vs24},={vs25},={vs26},={vs27},={vs28},={vs29},={vs30},={vs31},={v0},={v1},={v2},={v3},={v4},={v5},={v6},={v7},={v8},={v9},={v10},={v11},={v12},={v13},={v14},={v15},={v16},={v17},={v18},={v19},~{cr0},~{cr1},~{cr5},~{cr6},~{cr7},~{ctr},~{lr},~{xer}"()
#[no_mangle]
pub unsafe fn clobber_abi() {
    asm!("", clobber_abi("C"), options(nostack, nomem, preserves_flags));
}

// CHECK-LABEL: @clobber_no_preserves_flags
// CHECK: call void asm sideeffect "nop", ""()
#[no_mangle]
pub unsafe fn clobber_no_preserves_flags() {
    // Use a nop to prevent aliasing of identical functions here.
    asm!("nop", options(nostack, nomem));
}

// CHECK-LABEL: @cr0_clobber_no_preserves_flags
// CHECK: call void asm sideeffect "nop; nop", "~{cr0}"()
#[no_mangle]
pub unsafe fn cr0_clobber_no_preserves_flags() {
    // Use nop; nop to prevent aliasing of identical functions here.
    asm!("nop; nop", out("cr0") _, options(nostack, nomem));
}

// CHECK-LABEL: @clobber_preservesflags
// CHECK: call void asm sideeffect "", "~{memory}"()
#[no_mangle]
pub unsafe fn clobber_preservesflags() {
    asm!("", options(nostack, preserves_flags));
}

// Output format depends on the availability of altivec and vsx
// CHECK-LABEL: @clobber_abi_no_preserves_flags
#[no_mangle]
// powerpc: asm sideeffect "nop", "={r0},={r3},={r4},={r5},={r6},={r7},={r8},={r9},={r10},={r11},={r12},={f0},={f1},={f2},={f3},={f4},={f5},={f6},={f7},={f8},={f9},={f10},={f11},={f12},={f13},~{vs0},~{vs1},~{vs2},~{vs3},~{vs4},~{vs5},~{vs6},~{vs7},~{vs8},~{vs9},~{vs10},~{vs11},~{vs12},~{vs13},~{vs14},~{vs15},~{vs16},~{vs17},~{vs18},~{vs19},~{vs20},~{vs21},~{vs22},~{vs23},~{vs24},~{vs25},~{vs26},~{vs27},~{vs28},~{vs29},~{vs30},~{vs31},~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{cr0},~{cr1},~{cr5},~{cr6},~{cr7},~{ctr},~{lr},~{xer}"()
// powerpc64: asm sideeffect "nop", "={r0},={r3},={r4},={r5},={r6},={r7},={r8},={r9},={r10},={r11},={r12},={f0},={f1},={f2},={f3},={f4},={f5},={f6},={f7},={f8},={f9},={f10},={f11},={f12},={f13},={v0},={v1},={v2},={v3},={v4},={v5},={v6},={v7},={v8},={v9},={v10},={v11},={v12},={v13},={v14},={v15},={v16},={v17},={v18},={v19},~{vs0},~{vs1},~{vs2},~{vs3},~{vs4},~{vs5},~{vs6},~{vs7},~{vs8},~{vs9},~{vs10},~{vs11},~{vs12},~{vs13},~{vs14},~{vs15},~{vs16},~{vs17},~{vs18},~{vs19},~{vs20},~{vs21},~{vs22},~{vs23},~{vs24},~{vs25},~{vs26},~{vs27},~{vs28},~{vs29},~{vs30},~{vs31},~{cr0},~{cr1},~{cr5},~{cr6},~{cr7},~{ctr},~{lr},~{xer}"()
// powerpc64le: asm sideeffect "nop", "={r0},={r3},={r4},={r5},={r6},={r7},={r8},={r9},={r10},={r11},={r12},={f0},={f1},={f2},={f3},={f4},={f5},={f6},={f7},={f8},={f9},={f10},={f11},={f12},={f13},={vs0},={vs1},={vs2},={vs3},={vs4},={vs5},={vs6},={vs7},={vs8},={vs9},={vs10},={vs11},={vs12},={vs13},={vs14},={vs15},={vs16},={vs17},={vs18},={vs19},={vs20},={vs21},={vs22},={vs23},={vs24},={vs25},={vs26},={vs27},={vs28},={vs29},={vs30},={vs31},={v0},={v1},={v2},={v3},={v4},={v5},={v6},={v7},={v8},={v9},={v10},={v11},={v12},={v13},={v14},={v15},={v16},={v17},={v18},={v19},~{cr0},~{cr1},~{cr5},~{cr6},~{cr7},~{ctr},~{lr},~{xer}"()
// aix64: asm sideeffect "nop", "={r0},={r3},={r4},={r5},={r6},={r7},={r8},={r9},={r10},={r11},={r12},={f0},={f1},={f2},={f3},={f4},={f5},={f6},={f7},={f8},={f9},={f10},={f11},={f12},={f13},={vs0},={vs1},={vs2},={vs3},={vs4},={vs5},={vs6},={vs7},={vs8},={vs9},={vs10},={vs11},={vs12},={vs13},={vs14},={vs15},={vs16},={vs17},={vs18},={vs19},={vs20},={vs21},={vs22},={vs23},={vs24},={vs25},={vs26},={vs27},={vs28},={vs29},={vs30},={vs31},={v0},={v1},={v2},={v3},={v4},={v5},={v6},={v7},={v8},={v9},={v10},={v11},={v12},={v13},={v14},={v15},={v16},={v17},={v18},={v19},~{cr0},~{cr1},~{cr5},~{cr6},~{cr7},~{ctr},~{lr},~{xer}"()
pub unsafe fn clobber_abi_no_preserves_flags() {
    // Use a nop to prevent aliasing of identical functions here.
    asm!("nop", clobber_abi("C"), options(nostack, nomem));
}
