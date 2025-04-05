// Test that loads into registers x16..=x31 are never generated for riscv32{e,em,emc} targets
//
//@ add-core-stubs
//@ build-fail
//@ revisions: riscv32e riscv32em riscv32emc
//
//@ compile-flags: --crate-type=rlib
//@ [riscv32e] needs-llvm-components: riscv
//@ [riscv32e] compile-flags: --target=riscv32e-unknown-none-elf
//@ [riscv32em] needs-llvm-components: riscv
//@ [riscv32em] compile-flags: --target=riscv32em-unknown-none-elf
//@ [riscv32emc] needs-llvm-components: riscv
//@ [riscv32emc] compile-flags: --target=riscv32emc-unknown-none-elf

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
    //~^ ERROR invalid operand for instruction
    //~| NOTE instantiated into assembly here
    asm!("li x17, 0");
    //~^ ERROR invalid operand for instruction
    //~| NOTE instantiated into assembly here
    asm!("li x18, 0");
    //~^ ERROR invalid operand for instruction
    //~| NOTE instantiated into assembly here
    asm!("li x19, 0");
    //~^ ERROR invalid operand for instruction
    //~| NOTE instantiated into assembly here
    asm!("li x20, 0");
    //~^ ERROR invalid operand for instruction
    //~| NOTE instantiated into assembly here
    asm!("li x21, 0");
    //~^ ERROR invalid operand for instruction
    //~| NOTE instantiated into assembly here
    asm!("li x22, 0");
    //~^ ERROR invalid operand for instruction
    //~| NOTE instantiated into assembly here
    asm!("li x23, 0");
    //~^ ERROR invalid operand for instruction
    //~| NOTE instantiated into assembly here
    asm!("li x24, 0");
    //~^ ERROR invalid operand for instruction
    //~| NOTE instantiated into assembly here
    asm!("li x25, 0");
    //~^ ERROR invalid operand for instruction
    //~| NOTE instantiated into assembly here
    asm!("li x26, 0");
    //~^ ERROR invalid operand for instruction
    //~| NOTE instantiated into assembly here
    asm!("li x27, 0");
    //~^ ERROR invalid operand for instruction
    //~| NOTE instantiated into assembly here
    asm!("li x28, 0");
    //~^ ERROR invalid operand for instruction
    //~| NOTE instantiated into assembly here
    asm!("li x29, 0");
    //~^ ERROR invalid operand for instruction
    //~| NOTE instantiated into assembly here
    asm!("li x30, 0");
    //~^ ERROR invalid operand for instruction
    //~| NOTE instantiated into assembly here
    asm!("li x31, 0");
    //~^ ERROR invalid operand for instruction
    //~| NOTE instantiated into assembly here
}
