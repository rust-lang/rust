//@ edition:2021
//@ needs-asm-support

use std::arch::asm;

async unsafe fn foo<'a>() {
    asm!("/* {0} */", const N); //~ ERROR E0425
}

fn main() {}
