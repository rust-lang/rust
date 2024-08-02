//@ needs-asm-support
//@ ignore-nvptx64
//@ ignore-spirv

use std::arch::{asm, global_asm};

// Sym operands must point to a function or static

const C: i32 = 0;
static S: i32 = 0;
global_asm!("{}", sym S);
global_asm!("{}", sym main);
global_asm!("{}", sym C);
//~^ ERROR invalid `sym` operand

fn main() {
    unsafe {
        // Sym operands must point to a function or static

        let x: u64 = 0;
        const C: i32 = 0;
        static S: i32 = 0;
        asm!("{}", sym S);
        asm!("{}", sym main);
        asm!("{}", sym C);
        //~^ ERROR invalid `sym` operand
        asm!("{}", sym x);
        //~^ ERROR invalid `sym` operand
    }
}

unsafe fn generic<T>() {
    asm!("{}", sym generic::<T>);
}
