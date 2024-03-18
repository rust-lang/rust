//@ build-fail
#![feature(asm_const)]

use std::arch::global_asm;

fn main() {}

global_asm!("/* {} */", const 1 << 500); //~ ERROR evaluation of constant value failed [E0080]

global_asm!("/* {} */", const 1 / 0); //~ ERROR evaluation of constant value failed [E0080]
