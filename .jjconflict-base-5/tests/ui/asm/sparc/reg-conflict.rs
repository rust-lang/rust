//@ add-minicore
//@ revisions: sparc sparcv8plus sparc64
//@[sparc] compile-flags: --target sparc-unknown-none-elf
//@[sparc] needs-llvm-components: sparc
//@[sparcv8plus] compile-flags: --target sparc-unknown-linux-gnu
//@[sparcv8plus] needs-llvm-components: sparc
//@[sparc64] compile-flags: --target sparc64-unknown-linux-gnu
//@[sparc64] needs-llvm-components: sparc
//@ ignore-backends: gcc

#![crate_type = "rlib"]
#![feature(no_core, asm_experimental_arch, f128)]
#![no_core]

extern crate minicore;
use minicore::*;

fn f() {
    unsafe {
        asm!("", in("f6") 0.0_f32, in("d6") 0.0_f64);
        //~^ ERROR register `d6` conflicts with register `f6`
        asm!("", in("f7") 0.0_f32, in("d6") 0.0_f64);
        //~^ ERROR register `d6` conflicts with register `f7`
        asm!("", in("f8") 0.0_f32, in("q8") 0.0_f128);
        //~^ ERROR register `q8` conflicts with register `f8`
        asm!("", in("f9") 0.0_f32, in("q8") 0.0_f128);
        //~^ ERROR register `q8` conflicts with register `f9`
        asm!("", in("f10") 0.0_f32, in("q8") 0.0_f128);
        //~^ ERROR register `q8` conflicts with register `f10`
        asm!("", in("f11") 0.0_f32, in("q8") 0.0_f128);
        //~^ ERROR register `q8` conflicts with register `f11`
        asm!("", in("d12") 0.0_f64, in("q12") 0.0_f128);
        //~^ ERROR register `q12` conflicts with register `d12`
        asm!("", in("d12") 0.0_f64, in("q12") 0.0_f128);
        //~^ ERROR register `q12` conflicts with register `d12`
        asm!("", in("d14") 0.0_f64, in("q12") 0.0_f128);
        //~^ ERROR register `q12` conflicts with register `d14`
        asm!("", in("d14") 0.0_f64, in("q12") 0.0_f128);
        //~^ ERROR register `q12` conflicts with register `d14`
    }
}
