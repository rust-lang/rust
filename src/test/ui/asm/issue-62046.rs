// build-fail
// ignore-emscripten no asm! support

#![feature(asm)]

fn main() {
    unsafe {
        asm!("nop" : "+r"("r15"));
        //~^ malformed inline assembly
    }
}
