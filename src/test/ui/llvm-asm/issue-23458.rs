#![feature(llvm_asm)]
#![allow(deprecated)] // llvm_asm!
// compile-flags: -Ccodegen-units=1
// build-fail
// only-x86_64

fn main() {
    unsafe {
        llvm_asm!("int $3"); //~ ERROR too few operands for instruction
                             //~| ERROR invalid operand in inline asm
    }
}
