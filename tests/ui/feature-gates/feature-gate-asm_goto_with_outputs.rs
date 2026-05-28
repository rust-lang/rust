//@ only-x86_64

use std::arch::asm;

fn main() {
    let mut _out: u64;
    unsafe {
        asm!("mov {}, 1", "jmp {}", out(reg) _out, label {});
        //~^ ERROR using both label and output operands for inline assembly is unstable
    }
}
