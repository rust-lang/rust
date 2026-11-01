//@ needs-asm-support

use std::arch::asm;
use std::ptr::NonNull;

fn _f<T>() {
    let p: NonNull<T>;
    unsafe {
        asm!("/* {} */", out(reg) p);
        //~^ ERROR cannot use value of type `NonNull<T>` for inline assembly
    }
}

fn _g<T>(mut p: NonNull<T>) {
    unsafe {
        asm!("/* {} */", inout(reg) p);
        //~^ ERROR cannot use value of type `NonNull<T>` for inline assembly
    }
}

fn main() {}
