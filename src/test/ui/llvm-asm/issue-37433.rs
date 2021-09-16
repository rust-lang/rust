// build-fail
// ignore-emscripten no llvm_asm! support

#![feature(llvm_asm)]
#![allow(deprecated)] // llvm_asm!

fn main() {
    unsafe {
        llvm_asm!("" :: "r"(""));
        //~^ ERROR: invalid value for constraint in inline assembly
    }
}
