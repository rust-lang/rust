//@ needs-asm-support

use std::arch::asm;

fn main() {
    unsafe {
        asm!("", in("invalid") "".len());
        //~^ ERROR: invalid register `invalid`: unknown register
    }
}
