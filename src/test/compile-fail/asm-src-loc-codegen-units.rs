// WONTFIX(#20184) Needs landing pads (not present in stage1) or the compiler hangs.
// ignore-stage1
// compile-flags: -C codegen-units=2
// ignore-emscripten

#![feature(llvm_asm)]

fn main() {
    unsafe {
        llvm_asm!("nowayisthisavalidinstruction"); //~ ERROR instruction
    }
}
