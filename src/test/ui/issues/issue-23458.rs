#![feature(asm)]

// only-x86_64

fn main() {
    unsafe {
        asm!("int $3"); //~ ERROR too few operands for instruction
                        //~| ERROR invalid operand in inline asm
    }
}
