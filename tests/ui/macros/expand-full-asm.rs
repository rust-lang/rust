//@only-aarch64
//@check-pass
//@edition: 2018

// https://github.com/rust-lang/rust/issues/98291

use std::arch::{asm, global_asm};

macro_rules! wrap {
    () => {
        macro_rules! _a {
            () => {
                "nop"
            };
        }
    };
}

wrap!();

use _a as a;

fn main() {
    unsafe { asm!(a!()); }
}

global_asm!(a!());
