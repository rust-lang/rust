// ignore-emscripten

#![feature(asm)]

fn main() {
    let a; //~ ERROR type annotations needed
    asm!("nop" : "r"(a));
    //~^ ERROR E0661
}
