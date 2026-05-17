//@ needs-asm-support
//@ build-pass

#![feature(asm_interpolate)]

use std::arch::asm;

trait Foo {
    const STR: &str;
}

impl Foo for usize {
    const STR: &str = "usize";
}

fn main() {
    const TEST: &str = "test";

    unsafe {
        asm!("/* {0} */", interpolate usize::STR);
        asm!("/* {0} */", interpolate "test");
        asm!("/* {0} */", interpolate TEST);
    }
}
