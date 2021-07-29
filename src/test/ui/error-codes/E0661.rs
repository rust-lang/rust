// ignore-emscripten

#![feature(llvm_asm)]
#![allow(deprecated)] // llvm_asm!

fn main() {
    let a; //~ ERROR type annotations needed
    llvm_asm!("nop" : "r"(a));
    //~^ ERROR E0661
}
