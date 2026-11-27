//@ add-minicore
//@ revisions: mips32 mips64 mips32r6 mips64r6
//@[mips32] compile-flags: --target mips-unknown-linux-gnu
//@[mips32] needs-llvm-components: mips
//@[mips64] compile-flags: --target mips64-unknown-linux-gnuabi64
//@[mips64] needs-llvm-components: mips
//@[mips32r6] compile-flags: --target mipsisa32r6-unknown-linux-gnu
//@[mips32r6] needs-llvm-components: mips
//@[mips64r6] compile-flags: --target mipsisa64r6-unknown-linux-gnuabi64
//@[mips64r6] needs-llvm-components: mips
//@ ignore-backends: gcc

#![crate_type = "rlib"]
#![feature(no_core, asm_experimental_arch, f16)]
#![no_core]

extern crate minicore;
use minicore::*;

fn f() {
    unsafe {
        asm!("# {}", in(wreg) 0.0);
        //~^ ERROR register class `wreg` requires the `msa` target feature
        asm!("", in("$w10") 0.0);
        //~^ ERROR register class `wreg` requires the `msa` target feature
    }
}
