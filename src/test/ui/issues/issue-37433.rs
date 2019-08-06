// ignore-emscripten no asm! support

#![feature(asm)]

fn main() {
    unsafe {
        asm!("" :: "r"(""));
        //~^ ERROR: invalid value for constraint in inline assembly
    }
}
