// only-x86_64

use std::arch::asm;

fn bar<const N: usize>() {}

fn foo<const N: usize>() {
    unsafe {
        asm!("mov eax, {}", sym bar::<N>);
        //~^ ERROR sym operands for inline assembly are unstable
    }
}

fn main() {
    unsafe {
        asm!("mov eax, {}", sym foo::<0>);
        //~^ ERROR sym operands for inline assembly are unstable
    }
}
