// Ensure that we don't ICE when constructing the fake MIR body for a global
// asm when the body has errors. See #137470.

//@ needs-asm-support

use std::arch::global_asm;

global_asm!("/* {} */", sym a);
//~^ ERROR cannot find value `a` in this scope

fn main() {}
