// Test that loads into registers x16..=x31 are never generated for riscv32{e,em,emc} targets
//
//@ add-minicore
//@ build-fail
//@ revisions: riscv32e_llvm23 riscv32em_llvm23 riscv32emc_llvm23
//@ revisions: riscv32e_llvm24 riscv32em_llvm24 riscv32emc_llvm24
//@ compile-flags: --crate-type=rlib
//@ [riscv32e_llvm23] needs-llvm-components: riscv
//@ [riscv32e_llvm23] compile-flags: --target=riscv32e-unknown-none-elf
//@ [riscv32e_llvm23] max-llvm-major-version: 23
//@ [riscv32e_llvm24] needs-llvm-components: riscv
//@ [riscv32e_llvm24] compile-flags: --target=riscv32e-unknown-none-elf
//@ [riscv32e_llvm24] min-llvm-version: 24

//@ [riscv32em_llvm23] needs-llvm-components: riscv
//@ [riscv32em_llvm23] compile-flags: --target=riscv32em-unknown-none-elf
//@ [riscv32em_llvm23] max-llvm-major-version: 23
//@ [riscv32em_llvm24] needs-llvm-components: riscv
//@ [riscv32em_llvm24] compile-flags: --target=riscv32em-unknown-none-elf
//@ [riscv32em_llvm24] min-llvm-version: 24

//@ [riscv32emc_llvm23] needs-llvm-components: riscv
//@ [riscv32emc_llvm23] compile-flags: --target=riscv32emc-unknown-none-elf
//@ [riscv32emc_llvm23] max-llvm-major-version: 23
//@ [riscv32emc_llvm24] needs-llvm-components: riscv
//@ [riscv32emc_llvm24] compile-flags: --target=riscv32emc-unknown-none-elf
//@ [riscv32emc_llvm24] min-llvm-version: 24
//@ ignore-backends: gcc

// Unlike bad-reg.rs, this tests if the assembler can reject invalid registers
// usage in assembly code.

#![no_core]
#![feature(no_core)]

extern crate minicore;
use minicore::*;

// Verify registers x1..=x15 are addressable on riscv32e, but registers x16..=x31 are not
#[no_mangle]
pub unsafe fn registers() {
    asm!("li x1, 0");
    asm!("li x2, 0");
    asm!("li x3, 0");
    asm!("li x4, 0");
    asm!("li x5, 0");
    asm!("li x6, 0");
    asm!("li x7, 0");
    asm!("li x8, 0");
    asm!("li x9, 0");
    asm!("li x10, 0");
    asm!("li x11, 0");
    asm!("li x12, 0");
    asm!("li x13, 0");
    asm!("li x14, 0");
    asm!("li x15, 0");
    asm!("li x16, 0");
    //[riscv32e_llvm23,riscv32em_llvm23,riscv32emc_llvm23]~^ ERROR invalid operand for instruction
    //[riscv32e_llvm24,riscv32em_llvm24,riscv32emc_llvm24]~^^ ERROR register must be a GPR
    //~| NOTE instantiated into assembly here
    asm!("li x17, 0");
    //[riscv32e_llvm23,riscv32em_llvm23,riscv32emc_llvm23]~^ ERROR invalid operand for instruction
    //[riscv32e_llvm24,riscv32em_llvm24,riscv32emc_llvm24]~^^ ERROR register must be a GPR
    //~| NOTE instantiated into assembly here
    asm!("li x18, 0");
    //[riscv32e_llvm23,riscv32em_llvm23,riscv32emc_llvm23]~^ ERROR invalid operand for instruction
    //[riscv32e_llvm24,riscv32em_llvm24,riscv32emc_llvm24]~^^ ERROR register must be a GPR
    //~| NOTE instantiated into assembly here
    asm!("li x19, 0");
    //[riscv32e_llvm23,riscv32em_llvm23,riscv32emc_llvm23]~^ ERROR invalid operand for instruction
    //[riscv32e_llvm24,riscv32em_llvm24,riscv32emc_llvm24]~^^ ERROR register must be a GPR
    //~| NOTE instantiated into assembly here
    asm!("li x20, 0");
    //[riscv32e_llvm23,riscv32em_llvm23,riscv32emc_llvm23]~^ ERROR invalid operand for instruction
    //[riscv32e_llvm24,riscv32em_llvm24,riscv32emc_llvm24]~^^ ERROR register must be a GPR
    //~| NOTE instantiated into assembly here
    asm!("li x21, 0");
    //[riscv32e_llvm23,riscv32em_llvm23,riscv32emc_llvm23]~^ ERROR invalid operand for instruction
    //[riscv32e_llvm24,riscv32em_llvm24,riscv32emc_llvm24]~^^ ERROR register must be a GPR
    //~| NOTE instantiated into assembly here
    asm!("li x22, 0");
    //[riscv32e_llvm23,riscv32em_llvm23,riscv32emc_llvm23]~^ ERROR invalid operand for instruction
    //[riscv32e_llvm24,riscv32em_llvm24,riscv32emc_llvm24]~^^ ERROR register must be a GPR
    //~| NOTE instantiated into assembly here
    asm!("li x23, 0");
    //[riscv32e_llvm23,riscv32em_llvm23,riscv32emc_llvm23]~^ ERROR invalid operand for instruction
    //[riscv32e_llvm24,riscv32em_llvm24,riscv32emc_llvm24]~^^ ERROR register must be a GPR
    //~| NOTE instantiated into assembly here
    asm!("li x24, 0");
    //[riscv32e_llvm23,riscv32em_llvm23,riscv32emc_llvm23]~^ ERROR invalid operand for instruction
    //[riscv32e_llvm24,riscv32em_llvm24,riscv32emc_llvm24]~^^ ERROR register must be a GPR
    //~| NOTE instantiated into assembly here
    asm!("li x25, 0");
    //[riscv32e_llvm23,riscv32em_llvm23,riscv32emc_llvm23]~^ ERROR invalid operand for instruction
    //[riscv32e_llvm24,riscv32em_llvm24,riscv32emc_llvm24]~^^ ERROR register must be a GPR
    //~| NOTE instantiated into assembly here
    asm!("li x26, 0");
    //[riscv32e_llvm23,riscv32em_llvm23,riscv32emc_llvm23]~^ ERROR invalid operand for instruction
    //[riscv32e_llvm24,riscv32em_llvm24,riscv32emc_llvm24]~^^ ERROR register must be a GPR
    //~| NOTE instantiated into assembly here
    asm!("li x27, 0");
    //[riscv32e_llvm23,riscv32em_llvm23,riscv32emc_llvm23]~^ ERROR invalid operand for instruction
    //[riscv32e_llvm24,riscv32em_llvm24,riscv32emc_llvm24]~^^ ERROR register must be a GPR
    //~| NOTE instantiated into assembly here
    asm!("li x28, 0");
    //[riscv32e_llvm23,riscv32em_llvm23,riscv32emc_llvm23]~^ ERROR invalid operand for instruction
    //[riscv32e_llvm24,riscv32em_llvm24,riscv32emc_llvm24]~^^ ERROR register must be a GPR
    //~| NOTE instantiated into assembly here
    asm!("li x29, 0");
    //[riscv32e_llvm23,riscv32em_llvm23,riscv32emc_llvm23]~^ ERROR invalid operand for instruction
    //[riscv32e_llvm24,riscv32em_llvm24,riscv32emc_llvm24]~^^ ERROR register must be a GPR
    //~| NOTE instantiated into assembly here
    asm!("li x30, 0");
    //[riscv32e_llvm23,riscv32em_llvm23,riscv32emc_llvm23]~^ ERROR invalid operand for instruction
    //[riscv32e_llvm24,riscv32em_llvm24,riscv32emc_llvm24]~^^ ERROR register must be a GPR
    //~| NOTE instantiated into assembly here
    asm!("li x31, 0");
    //[riscv32e_llvm23,riscv32em_llvm23,riscv32emc_llvm23]~^ ERROR invalid operand for instruction
    //[riscv32e_llvm24,riscv32em_llvm24,riscv32emc_llvm24]~^^ ERROR register must be a GPR
    //~| NOTE instantiated into assembly here
}
