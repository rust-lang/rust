// needs-asm-support
// ignore-nvptx64
// ignore-spirv
// ignore-wasm32

use std::arch::asm;

fn main() {
    unsafe {
        // Can't output to borrowed values.

        let mut a = 0isize;
        let p = &a;
        asm!("{}", out(reg) a);
        //~^ cannot assign to `a` because it is borrowed
        println!("{}", p);

        // Can't read from mutable borrowed values.

        let mut a = 0isize;
        let p = &mut a;
        asm!("{}", in(reg) a);
        //~^ cannot use `a` because it was mutably borrowed
        println!("{}", p);
    }
}
