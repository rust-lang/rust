//@ needs-asm-support

#![feature(asm_interpolate)]

use std::arch::asm;

fn main() {
    let x = "";

    unsafe {
        asm!("/* {0} */", interpolate x);
        //~^ ERROR attempt to use a non-constant value in a constant [E0435]
    }
}
