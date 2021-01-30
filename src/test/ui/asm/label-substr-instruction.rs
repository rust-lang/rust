// compile-flags: -g
// build-pass

#![feature(asm)]

fn main() {
    unsafe {
        asm!("jmp j", "nop", "j: nop");
    }
}
