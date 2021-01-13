// build-fail
// dont-check-compiler-stderr
// ignore-emscripten

#![feature(llvm_asm)]

fn main() {
    unsafe {
        llvm_asm!("nowayisthisavalidinstruction"); //~ ERROR instruction
    }
}
