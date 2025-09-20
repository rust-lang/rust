//@ add-core-stubs
//@ needs-asm-support
//@ revisions: loongarch32_ilp32d loongarch32_ilp32s loongarch64_lp64d loongarch64_lp64s
//@[loongarch32_ilp32d] compile-flags: --target loongarch32-unknown-none
//@[loongarch32_ilp32d] needs-llvm-components: loongarch
//@[loongarch32_ilp32s] compile-flags: --target loongarch32-unknown-none-softfloat
//@[loongarch32_ilp32s] needs-llvm-components: loongarch
//@[loongarch64_lp64d] compile-flags: --target loongarch64-unknown-linux-gnu
//@[loongarch64_lp64d] needs-llvm-components: loongarch
//@[loongarch64_lp64s] compile-flags: --target loongarch64-unknown-none-softfloat
//@[loongarch64_lp64s] needs-llvm-components: loongarch

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
        asm!("", out("$r0") _);
        //~^ ERROR constant zero cannot be used as an operand for inline asm
        asm!("", out("$tp") _);
        //~^ ERROR invalid register `$tp`: reserved for TLS
        asm!("", out("$sp") _);
        //~^ ERROR invalid register `$sp`: the stack pointer cannot be used as an operand for inline asm
        asm!("", out("$r21") _);
        //~^ ERROR invalid register `$r21`: reserved by the ABI
        asm!("", out("$fp") _);
        //~^ ERROR invalid register `$fp`: the frame pointer cannot be used as an operand for inline asm
        asm!("", out("$r31") _);
        //~^ ERROR invalid register `$r31`: $r31 is used internally by LLVM and cannot be used as an operand for inline asm

        asm!("", out("$f0") _); // ok
        asm!("/* {} */", in(freg) f);
        //[loongarch32_ilp32s,loongarch64_lp64s]~^ ERROR register class `freg` requires at least one of the following target features: d, f
        asm!("/* {} */", out(freg) _);
        //[loongarch32_ilp32s,loongarch64_lp64s]~^ ERROR register class `freg` requires at least one of the following target features: d, f
        asm!("/* {} */", in(freg) d);
        //[loongarch32_ilp32s,loongarch64_lp64s]~^ ERROR register class `freg` requires at least one of the following target features: d, f
        asm!("/* {} */", out(freg) d);
        //[loongarch32_ilp32s,loongarch64_lp64s]~^ ERROR register class `freg` requires at least one of the following target features: d, f
    }
}
