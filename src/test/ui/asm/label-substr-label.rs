// compile-flags: -g
// build-pass

#![feature(asm)]

fn main() {
    unsafe {
        asm!("jmp l", "l: nop", "jmp l2", "l2: nop");
    }
}
