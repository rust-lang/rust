// ignore-emscripten no asm! support

#![feature(asm)]

fn main() {
    unsafe {
        asm! {"mov $0,$1"::"0"("bx"),"1"(0x00)}
        //~^ ERROR: invalid value for constraint in inline assembly
    }
}
