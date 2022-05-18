// only-x86_64

use std::arch::asm;

unsafe fn foo<const N: usize>() {
    asm!("mov eax, {}", const N + 1);
    //~^ ERROR const operands for inline assembly are unstable
}

fn main() {
    unsafe {
        foo::<0>();
        asm!("mov eax, {}", const 123);
        //~^ ERROR const operands for inline assembly are unstable
    }
}
