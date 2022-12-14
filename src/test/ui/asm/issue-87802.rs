// needs-asm-support
// ignore-nvptx64
// ignore-spirv
// ignore-wasm32
// Make sure rustc doesn't ICE on asm! when output type is !.

use std::arch::asm;

fn hmm() -> ! {
    let x;
    unsafe {
        asm!("/* {0} */", out(reg) x);
        //~^ ERROR cannot use value of type `!` for inline assembly
    }
    x
}

fn main() {
    hmm();
}
