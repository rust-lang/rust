//@ add-core-stubs
//@ revisions: sparc sparcv8plus sparc64
//@[sparc] compile-flags: --target sparc-unknown-none-elf
//@[sparc] needs-llvm-components: sparc
//@[sparcv8plus] compile-flags: --target sparc-unknown-linux-gnu
//@[sparcv8plus] needs-llvm-components: sparc
//@[sparc64] compile-flags: --target sparc64-unknown-linux-gnu
//@[sparc64] needs-llvm-components: sparc
//@ needs-asm-support

#![crate_type = "rlib"]
#![feature(no_core, asm_experimental_arch)]
#![no_core]

extern crate minicore;
use minicore::*;

fn f() {
    let mut x = 0;
    unsafe {
        // Unsupported registers
        asm!("", out("g0") _);
        //~^ ERROR invalid register `g0`: g0 is always zero and cannot be used as an operand for inline asm
        // FIXME: see FIXME in compiler/rustc_target/src/asm/sparc.rs.
        asm!("", out("g1") _);
        //~^ ERROR invalid register `g1`: reserved by LLVM and cannot be used as an operand for inline asm
        asm!("", out("g2") _);
        asm!("", out("g3") _);
        asm!("", out("g4") _);
        asm!("", out("g5") _);
        //[sparc,sparcv8plus]~^ ERROR cannot use register `r5`: g5 is reserved for system on SPARC32
        asm!("", out("g6") _);
        //~^ ERROR invalid register `g6`: reserved for system and cannot be used as an operand for inline asm
        asm!("", out("g7") _);
        //~^ ERROR invalid register `g7`: reserved for system and cannot be used as an operand for inline asm
        asm!("", out("sp") _);
        //~^ ERROR invalid register `sp`: the stack pointer cannot be used as an operand for inline asm
        asm!("", out("fp") _);
        //~^ ERROR invalid register `fp`: the frame pointer cannot be used as an operand for inline asm
        asm!("", out("i7") _);
        //~^ ERROR invalid register `i7`: the return address register cannot be used as an operand for inline asm

        // Clobber-only registers
        // yreg
        asm!("", out("y") _); // ok
        asm!("", in("y") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("", out("y") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", in(yreg) x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", out(yreg) _);
        //~^ ERROR can only be used as a clobber
    }
}
