//@ add-core-stubs
//@ revisions: hard soft
//@ assembly-output: emit-asm
//@ [hard] compile-flags: --target thumbv8m.main-none-eabihf --crate-type lib -Copt-level=1
//@ [soft] compile-flags: --target thumbv8m.main-none-eabi --crate-type lib -Copt-level=1
//@ [hard] needs-llvm-components: arm
//@ [soft] needs-llvm-components: arm
#![crate_type = "lib"]
#![feature(abi_cmse_nonsecure_call, cmse_nonsecure_entry, no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: __acle_se_entry_point:
// CHECK-NEXT: entry_point:
//
// Write return argument (two registers since 64bit integer)
// CHECK: movs r0, #0
// CHECK: movs r1, #0
//
// If we are using hard-float:
// * Check if the float registers were touched (bit 3 in CONTROL)
// hard: mrs     [[REG:r[0-9]+]], control
// hard: tst.w   [[REG]], #8
// hard: beq     [[LABEL:[\.a-zA-Z0-9_]+]]
//
// * If touched clear all float registers (d0..=d7)
// hard: vmov    d0,
// hard: vmov    d1,
// hard: vmov    d2,
// hard: vmov    d3,
// hard: vmov    d4,
// hard: vmov    d5,
// hard: vmov    d6,
// hard: vmov    d7,
//
// * If touched clear FPU status register
// hard: vmrs    [[REG:r[0-9]+]], fpscr
// hard: bic     [[REG]], [[REG]], #159
// hard: bic     [[REG]], [[REG]], #4026531840
// hard: vmsr    fpscr, [[REG]]
// hard: [[LABEL]]:
//
// Clear all other registers that might have been used
// CHECK: mov r2,
// CHECK: mov r3,
// CHECK: mov r12,
//
// Clear the flags
// CHECK: msr apsr_nzcvq,
//
// Branch back to non-secure side
// CHECK: bxns lr
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn entry_point() -> i64 {
    0
}

// NOTE for future codegen changes:
// The specific register assignment is not important, however:
// * all registers must be cleared before `blxns` is executed
//     (either by writing arguments or any other value)
// * the lowest bit on the address of the callee must be cleared
// * the flags need to be overwritten
// * `blxns` needs to be called with the callee address
//     (with the lowest bit cleared)
//
// CHECK-LABEL: call_nonsecure
// Save callee pointer
// CHECK: mov r12, r0
//
// All arguments are written to (writes r0..=r3)
// CHECK: movs r0, #0
// CHECK: movs r1, #1
// CHECK: movs r2, #2
// CHECK: movs r3, #3
//
// Lowest bit gets cleared on callee address
// CHECK: bic r12, r12, #1
//
// Ununsed registers get cleared (r4..=r11)
// CHECK: mov r4,
// CHECK: mov r5,
// CHECK: mov r6,
// CHECK: mov r7,
// CHECK: mov r8,
// CHECK: mov r9,
// CHECK: mov r10,
// CHECK: mov r11,
//
// Flags get cleared
// CHECK: msr apsr_nzcvq,
//
// Call to non-secure
// CHECK: blxns r12
#[no_mangle]
pub fn call_nonsecure(f: unsafe extern "cmse-nonsecure-call" fn(u32, u32, u32, u32) -> u64) -> u64 {
    unsafe { f(0, 1, 2, 3) }
}
