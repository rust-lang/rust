//@ add-minicore
//@ revisions: mips32 mips64 mips32r6 mips64r6
//@[mips32] compile-flags: --target mips-unknown-linux-gnu -Ctarget-feature=+mips32r5
//@[mips32] needs-llvm-components: mips
//@[mips64] compile-flags: --target mips64-unknown-linux-gnuabi64 -Ctarget-feature=+mips64r5
//@[mips64] needs-llvm-components: mips
//@[mips32r6] compile-flags: --target mipsisa32r6-unknown-linux-gnu
//@[mips32r6] needs-llvm-components: mips
//@[mips64r6] compile-flags: --target mipsisa64r6-unknown-linux-gnuabi64
//@[mips64r6] needs-llvm-components: mips
//@ compile-flags: -Ctarget-feature=+fp64,+msa
//@ ignore-backends: gcc

#![crate_type = "rlib"]
#![feature(no_core, asm_experimental_arch, f16)]
#![no_core]

//[mips32]~? WARN unknown and unstable feature specified for `-Ctarget-feature`: `mips32r5`
//[mips64]~? WARN unknown and unstable feature specified for `-Ctarget-feature`: `mips64r5`
//~? WARN unstable feature specified for `-Ctarget-feature`: `fp64`
//~? WARN unstable feature specified for `-Ctarget-feature`: `msa`

extern crate minicore;
use minicore::*;

fn f() {
    unsafe {
        asm!("", in("$w4") 0.0, in("$f4") 0.0);
        //~^ ERROR register `$f4` conflicts with register `$w4`
        asm!("", in("$w25") 0.0, in("$f25") 0.0);
        //~^ ERROR register `$f25` conflicts with register `$w25`
    }
}
