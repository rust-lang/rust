//@ add-core-stubs
//@ revisions: avr avrtiny
//@[avr] compile-flags: --target avr-none -C target-cpu=atmega328p
//@[avr] needs-llvm-components: avr
//@[avrtiny] compile-flags: --target avr-none -C target-cpu=attiny104
//@[avrtiny] needs-llvm-components: avr
//@ needs-asm-support
// ignore-tidy-linelength

#![crate_type = "rlib"]
#![feature(no_core, asm_experimental_arch)]
#![no_core]

extern crate minicore;
use minicore::*;

fn f() {
    unsafe {
        // Unsupported registers
        asm!("", out("Y") _);
        //~^ ERROR the frame pointer cannot be used as an operand for inline asm
        asm!("", out("YL") _);
        //~^ ERROR the frame pointer cannot be used as an operand for inline asm
        asm!("", out("YH") _);
        //~^ ERROR the frame pointer cannot be used as an operand for inline asm
        asm!("", out("SP") _);
        //~^ ERROR the stack pointer cannot be used as an operand for inline asm
        asm!("", out("SPL") _);
        //~^ ERROR the stack pointer cannot be used as an operand for inline asm
        asm!("", out("SPH") _);
        //~^ ERROR the stack pointer cannot be used as an operand for inline asm
        asm!("", out("r0") _);
        //~^ ERROR LLVM reserves r0 (scratch register) and r1 (zero register)
        asm!("", out("r1") _);
        //~^ ERROR LLVM reserves r0 (scratch register) and r1 (zero register)
        asm!("", out("r1r0") _);
        //~^ ERROR LLVM reserves r0 (scratch register) and r1 (zero register)

        // Unsupported only on AVRTiny
        asm!("", out("r2") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r3") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r4") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r5") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r6") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r7") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r8") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r9") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r10") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r11") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r12") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r13") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r14") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r15") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r16") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r17") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r3r2") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r5r4") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r7r6") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r9r8") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r11r10") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r13r12") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r15r14") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
        asm!("", out("r17r16") _);
        //[avrtiny]~^ ERROR on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM
    }
}
