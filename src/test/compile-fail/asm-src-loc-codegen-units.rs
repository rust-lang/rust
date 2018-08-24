// WONTFIX(#20184) Needs landing pads (not present in stage1) or the compiler hangs.
// ignore-stage1
// compile-flags: -C codegen-units=2
// ignore-emscripten

#![feature(asm)]

fn main() {
    unsafe {
        asm!("nowayisthisavalidinstruction"); //~ ERROR instruction
    }
}
