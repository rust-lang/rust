//@ add-core-stubs
//@ assembly-output: emit-asm
//@ revisions: rv32i rv64i rv32e
//@[rv32i] compile-flags: --target riscv32i-unknown-none-elf
//@[rv32i] needs-llvm-components: riscv
//@[rv64i] compile-flags: --target riscv64imac-unknown-none-elf
//@[rv64i] needs-llvm-components: riscv
//@[rv32e] compile-flags: --target riscv32e-unknown-none-elf
//@[rv32e] needs-llvm-components: riscv
// ignore-tidy-linelength

#![crate_type = "rlib"]
#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: @flags_clobber
// CHECK: call void asm sideeffect "", "~{fflags},~{vtype},~{vl},~{vxsat},~{vxrm}"()
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
// rv32i: asm sideeffect "", "={x1},={x5},={x6},={x7},={x10},={x11},={x12},={x13},={x14},={x15},={x16},={x17},={x28},={x29},={x30},={x31},~{f0},~{f1},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15},~{f16},~{f17},~{f28},~{f29},~{f30},~{f31},~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
// rv64i: asm sideeffect "", "={x1},={x5},={x6},={x7},={x10},={x11},={x12},={x13},={x14},={x15},={x16},={x17},={x28},={x29},={x30},={x31},~{f0},~{f1},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15},~{f16},~{f17},~{f28},~{f29},~{f30},~{f31},~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
// rv32e: asm sideeffect "", "={x1},={x5},={x6},={x7},={x10},={x11},={x12},={x13},={x14},={x15},~{f0},~{f1},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15},~{f16},~{f17},~{f28},~{f29},~{f30},~{f31},~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
#[no_mangle]
pub unsafe fn clobber_abi() {
    asm!("", clobber_abi("C"), options(nostack, nomem, preserves_flags));
}
