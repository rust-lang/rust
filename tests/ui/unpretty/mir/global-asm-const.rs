//@ compile-flags: -Zdump-mir=all --crate-type=lib
//@ needs-asm-support
//@ check-pass

use std::arch::{asm, global_asm};

// test that pretty mir printing of `const` operands in `global_asm!` does not ICE

global_asm!("/* {} */", const 5);

pub fn foo() {
    unsafe { asm!("/* {} */", const 5) };
}
