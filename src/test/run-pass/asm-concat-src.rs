// pretty-expanded FIXME #23616
// ignore-emscripten no asm

#![feature(asm)]

pub fn main() {
    unsafe { asm!(concat!("", "")) };
}
