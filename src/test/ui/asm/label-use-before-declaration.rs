// compile-flags: -g
// build-pass

#![feature(asm)]

fn main() {
    unsafe {
        asm!("jmp l", "nop", "l: nop");
    }
}
