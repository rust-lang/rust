// compile-flags: -g
// build-pass

#![feature(asm)]

#[inline(always)]
fn asm() {
    unsafe {
        asm!("duplabel: nop",);
    }
}

fn main() {
    asm();
    asm();
}
