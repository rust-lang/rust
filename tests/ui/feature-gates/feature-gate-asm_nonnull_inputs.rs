//@ only-x86_64

use std::arch::asm;

fn main() {
    let mut p = core::ptr::NonNull::<u32>::dangling();
    unsafe {
        asm!("/* {} */", in(reg) p);
        //~^ ERROR cannot use value of type `NonNull<u32>` for inline assembly
    }
}
