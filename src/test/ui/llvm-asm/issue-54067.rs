// check-pass
// ignore-emscripten no llvm_asm! support

#![feature(llvm_asm)]

pub fn boot(addr: Option<u32>) {
    unsafe {
        llvm_asm!("mov sp, $0"::"r" (addr));
    }
}

fn main() {}
