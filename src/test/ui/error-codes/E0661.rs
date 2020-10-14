// ignore-emscripten

#![feature(llvm_asm)]

fn main() {
    let a;
    llvm_asm!("nop" : "r"(a));
    //~^ ERROR E0661
}
