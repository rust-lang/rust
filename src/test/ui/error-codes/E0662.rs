// ignore-emscripten

#![feature(llvm_asm)]
#![allow(deprecated)] // llvm_asm!

fn main() {
    llvm_asm!("xor %eax, %eax"
              :
              : "=test"("a") //~ ERROR E0662
             );
}
