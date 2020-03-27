#![feature(llvm_asm)]

// build-fail
// only-x86_64

fn main() {
    unsafe {
        llvm_asm!("int $3"); //~ ERROR too few operands for instruction
                             //~| ERROR invalid operand in inline asm
    }
}
