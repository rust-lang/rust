//@ add-minicore
//@ compile-flags: --target hexagon-unknown-linux-musl -C target-feature=+hvx-length128b
//@ needs-llvm-components: hexagon
//@ ignore-backends: gcc

//~? WARN unstable feature specified for `-Ctarget-feature`: `hvx-length128b`

#![crate_type = "lib"]
#![feature(no_core, asm_experimental_arch)]
#![no_core]

extern crate minicore;
use minicore::*;

fn f() {
    let mut x: i32 = 0;
    let mut y: i64 = 0;
    unsafe {
        // Blocked registers
        asm!("", out("r19") _);
        //~^ ERROR invalid register `r19`: r19 is used internally by LLVM and cannot be used as an operand for inline asm
        asm!("", out("sp") _);
        //~^ ERROR invalid register `sp`: the stack pointer cannot be used as an operand for inline asm
        asm!("", out("r29") _);
        //~^ ERROR invalid register `r29`: the stack pointer cannot be used as an operand for inline asm
        asm!("", out("r30") _);
        //~^ ERROR invalid register `r30`: the frame register cannot be used as an operand for inline asm
        asm!("", out("fr") _);
        //~^ ERROR invalid register `fr`: the frame register cannot be used as an operand for inline asm
        asm!("", out("r31") _);
        //~^ ERROR invalid register `r31`: the link register cannot be used as an operand for inline asm
        asm!("", out("lr") _);
        //~^ ERROR invalid register `lr`: the link register cannot be used as an operand for inline asm

        // Blocked register pairs
        asm!("", out("r19:18") _);
        //~^ ERROR invalid register `r19:18`: r19 is used internally by LLVM and cannot be used as an operand for inline asm
        asm!("", out("r29:28") _);
        //~^ ERROR invalid register `r29:28`: the stack pointer cannot be used as an operand for inline asm
        asm!("", out("r31:30") _);
        //~^ ERROR invalid register `r31:30`: the frame register and link register cannot be used as an operand for inline asm

        // Clobber-only: preg
        asm!("", out("p0") _); // ok (clobber)
        asm!("", in("p0") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("", out("p0") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class

        // Clobber-only: qreg
        asm!("", out("q0") _); // ok (clobber)
        asm!("", in("q0") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("", out("q0") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class

        // Type mismatches: reg (supports i8, i16, i32, f32)
        asm!("/* {} */", in(reg) y);
        //~^ ERROR type `i64` cannot be used with this register class

        // Type mismatches: reg_pair (supports i64, f64)
        asm!("/* {} */", in(reg_pair) x);
        //~^ ERROR type `i32` cannot be used with this register class

        // Valid usage: reg
        asm!("", out("r0") _);
        asm!("/* {} */", in(reg) x);

        // Valid usage: reg_pair
        asm!("", out("r1:0") _);
        asm!("/* {} */", in(reg_pair) y);

        // Valid usage: vreg clobber
        asm!("", out("v0") _);

        // Valid usage: qreg clobber
        asm!("", out("q0") _);

        // Register pair overlap: r0 and r1:0 conflict
        asm!("", out("r0") _, out("r1:0") _);
        //~^ ERROR register `r1:0` conflicts with register `r0`
        asm!("", out("r1") _, out("r1:0") _);
        //~^ ERROR register `r1:0` conflicts with register `r1`

        // Non-overlapping pair and single: no conflict
        asm!("", out("r0") _, out("r3:2") _);
    }
}
