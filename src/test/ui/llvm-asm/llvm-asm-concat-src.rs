// run-pass
// pretty-expanded FIXME #23616
// ignore-emscripten no asm

#![feature(llvm_asm)]

pub fn main() {
    unsafe { llvm_asm!(concat!("", "")) };
}
