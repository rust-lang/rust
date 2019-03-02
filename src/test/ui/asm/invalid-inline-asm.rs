// ignore-wasm
// ignore-emscripten

#![feature(asm)]

fn main() {
    let byte = 0;
    let port = 0x80;

    unsafe { asm!("out %al, %dx" :: "a" (byte), "d" (port) :: "volatile"); }
    //~^ ERROR couldn't allocate input reg for constraint 'a'
}
