// build-fail
// ignore-emscripten no asm! support
// Regression test for #69092

#![feature(llvm_asm)]

fn main() {
    unsafe { llvm_asm!(".ascii \"Xen\0\""); }
    //~^ ERROR: expected string in '.ascii' directive
}
