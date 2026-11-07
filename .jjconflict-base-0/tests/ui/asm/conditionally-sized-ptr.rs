//@ check-pass
//@ needs-asm-support

use std::arch::asm;

fn _f<T>(p: *mut T) {
    unsafe {
        asm!("/* {} */", in(reg) p);
    }
}

fn main() {}
