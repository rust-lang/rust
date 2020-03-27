// build-fail
// ignore-emscripten no llvm_asm! support

#![feature(llvm_asm)]

fn main() {
    unsafe {
        llvm_asm! {"mov $0,$1"::"0"("bx"),"1"(0x00)}
        //~^ ERROR: invalid value for constraint in inline assembly
    }
}
