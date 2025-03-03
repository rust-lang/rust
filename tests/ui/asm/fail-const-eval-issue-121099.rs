//@ build-fail
//@ needs-asm-support

use std::arch::global_asm;

fn main() {}

global_asm!("/* {} */", const 1 << 500); //~ ERROR E0080

global_asm!("/* {} */", const 1 / 0); //~ ERROR E0080
