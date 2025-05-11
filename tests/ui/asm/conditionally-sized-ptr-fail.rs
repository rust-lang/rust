//@ needs-asm-support

use std::arch::asm;

fn _f<T: ?Sized>(p: *mut T) {
    unsafe {
        asm!("/* {} */", in(reg) p);
        //~^ ERROR cannot use value of unsized pointer type `*mut T` for inline assembly
    }
}

fn _g(p: *mut [u8]) {
    unsafe {
        asm!("/* {} */", in(reg) p);
        //~^ ERROR cannot use value of unsized pointer type `*mut [u8]` for inline assembly
    }
}

fn main() {}
