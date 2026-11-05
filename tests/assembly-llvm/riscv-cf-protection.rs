// Test that `-Z cf-protection=full` emits landing pads and the hardware shadow stack on RISC-V.

//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3 -Z cf-protection=full
//@ revisions: rv64 rv32 rva23
//@ needs-llvm-components: riscv
// LLVM 22 prints the shadow stack instructions as `mop.rr.*` unless Zicfiss is enabled.
//@ min-llvm-version: 23
//@ [rv64] compile-flags: --target riscv64gc-unknown-none-elf -Ctarget-feature=+zimop
//@ [rv32] compile-flags: --target riscv32imc-unknown-none-elf -Ctarget-feature=+zimop
//@ [rva23] compile-flags: --target riscv64a23-unknown-linux-gnu

#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

// Landing pads must be 4-byte aligned, even with compressed instructions.
// CHECK: .p2align 2
// CHECK-NEXT: .type call_twice,@function
// CHECK-NEXT: call_twice:
// CHECK-NEXT: .cfi_startproc
// CHECK-NEXT: lpad 0
// CHECK-NEXT: sspush ra
// CHECK: sspopchk ra
#[no_mangle]
pub fn call_twice(f: extern "C" fn()) {
    f();
    f();
}
