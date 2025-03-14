//@ add-core-stubs
//@ needs-asm-support
//@ revisions: riscv32i riscv32imafc riscv32gc riscv32e riscv64imac riscv64gc
//@[riscv32i] compile-flags: --target riscv32i-unknown-none-elf
//@[riscv32i] needs-llvm-components: riscv
//@[riscv32imafc] compile-flags: --target riscv32imafc-unknown-none-elf
//@[riscv32imafc] needs-llvm-components: riscv
//@[riscv32gc] compile-flags: --target riscv32gc-unknown-linux-gnu
//@[riscv32gc] needs-llvm-components: riscv
//@[riscv32e] compile-flags: --target riscv32e-unknown-none-elf
//@[riscv32e] needs-llvm-components: riscv
//@[riscv64imac] compile-flags: --target riscv64imac-unknown-none-elf
//@[riscv64imac] needs-llvm-components: riscv
//@[riscv64gc] compile-flags: --target riscv64gc-unknown-linux-gnu
//@[riscv64gc] needs-llvm-components: riscv

// Unlike riscv32e-registers.rs, this tests if the rustc can reject invalid registers
// usage in the asm! API (in, out, inout, etc.).

#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

fn f() {
    let mut x = 0;
    let mut f = 0.0_f32;
    let mut d = 0.0_f64;
    unsafe {
        // Unsupported registers
        asm!("", out("s1") _);
        //~^ ERROR invalid register `s1`: s1 is used internally by LLVM and cannot be used as an operand for inline asm
        asm!("", out("fp") _);
        //~^ ERROR invalid register `fp`: the frame pointer cannot be used as an operand for inline asm
        asm!("", out("sp") _);
        //~^ ERROR invalid register `sp`: the stack pointer cannot be used as an operand for inline asm
        asm!("", out("gp") _);
        //~^ ERROR invalid register `gp`: the global pointer cannot be used as an operand for inline asm
        asm!("", out("tp") _);
        //~^ ERROR invalid register `tp`: the thread pointer cannot be used as an operand for inline asm
        asm!("", out("zero") _);
        //~^ ERROR invalid register `zero`: the zero register cannot be used as an operand for inline asm

        asm!("", out("x16") _);
        //[riscv32e]~^ ERROR register can't be used with the `e` target feature
        asm!("", out("x17") _);
        //[riscv32e]~^ ERROR register can't be used with the `e` target feature
        asm!("", out("x18") _);
        //[riscv32e]~^ ERROR register can't be used with the `e` target feature
        asm!("", out("x19") _);
        //[riscv32e]~^ ERROR register can't be used with the `e` target feature
        asm!("", out("x20") _);
        //[riscv32e]~^ ERROR register can't be used with the `e` target feature
        asm!("", out("x21") _);
        //[riscv32e]~^ ERROR register can't be used with the `e` target feature
        asm!("", out("x22") _);
        //[riscv32e]~^ ERROR register can't be used with the `e` target feature
        asm!("", out("x23") _);
        //[riscv32e]~^ ERROR register can't be used with the `e` target feature
        asm!("", out("x24") _);
        //[riscv32e]~^ ERROR register can't be used with the `e` target feature
        asm!("", out("x25") _);
        //[riscv32e]~^ ERROR register can't be used with the `e` target feature
        asm!("", out("x26") _);
        //[riscv32e]~^ ERROR register can't be used with the `e` target feature
        asm!("", out("x27") _);
        //[riscv32e]~^ ERROR register can't be used with the `e` target feature
        asm!("", out("x28") _);
        //[riscv32e]~^ ERROR register can't be used with the `e` target feature
        asm!("", out("x29") _);
        //[riscv32e]~^ ERROR register can't be used with the `e` target feature
        asm!("", out("x30") _);
        //[riscv32e]~^ ERROR register can't be used with the `e` target feature
        asm!("", out("x31") _);
        //[riscv32e]~^ ERROR register can't be used with the `e` target feature

        asm!("", out("f0") _); // ok
        asm!("/* {} */", in(freg) f);
        //[riscv32i,riscv32e,riscv64imac]~^ ERROR register class `freg` requires at least one of the following target features: d, f
        asm!("/* {} */", out(freg) _);
        //[riscv32i,riscv32e,riscv64imac]~^ ERROR register class `freg` requires at least one of the following target features: d, f
        asm!("/* {} */", in(freg) d);
        //[riscv32i,riscv32e,riscv64imac]~^ ERROR register class `freg` requires at least one of the following target features: d, f
        //[riscv32imafc]~^^ ERROR `d` target feature is not enabled
        asm!("/* {} */", out(freg) d);
        //[riscv32i,riscv32e,riscv64imac]~^ ERROR register class `freg` requires at least one of the following target features: d, f
        //[riscv32imafc]~^^ ERROR `d` target feature is not enabled

        // Clobber-only registers
        // vreg
        asm!("", out("v0") _); // ok
        asm!("", in("v0") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("", out("v0") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", in(vreg) x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", out(vreg) _);
        //~^ ERROR can only be used as a clobber
    }
}
