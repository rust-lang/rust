// ignore-emscripten

#![feature(asm)]

fn main() {
    unsafe {
        asm!("nowayisthisavalidinstruction"); //~ ERROR instruction
    }
}
