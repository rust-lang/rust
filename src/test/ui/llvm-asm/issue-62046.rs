// build-fail
// ignore-emscripten no asm! support

#![feature(llvm_asm)]

fn main() {
    unsafe {
        llvm_asm!("nop" : "+r"("r15"));
        //~^ malformed inline assembly
    }
}
