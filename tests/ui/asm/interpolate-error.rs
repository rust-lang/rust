//@ needs-asm-support

#![feature(asm_interpolate)]

use std::arch::asm;

fn main() {
    unsafe {
        asm!("/* {0} */", interpolate 42);
        //~^ ERROR invalid type for `interpolate` operand
        asm!("/* {0} */", interpolate String::new());
        //~^ ERROR invalid type for `interpolate` operand
        asm!("/* {0} */", interpolate &String::new());
        //~^ ERROR invalid type for `interpolate` operand
    }
}
