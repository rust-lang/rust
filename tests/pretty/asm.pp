#![feature(prelude_import)]
#![no_std]
#[macro_use]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;
//@ pretty-mode:expanded
//@ pp-exact:asm.pp
//@ only-x86_64

use std::arch::asm;

pub fn main() {
    let a: i32;
    let mut b = 4i32;
    unsafe {
        asm!("");
        asm!("");
        asm!("", options(nomem, nostack));
        asm!("{0}", in(reg) 4);
        asm!("{0}", out(reg) a);
        asm!("{0}", inout(reg) b);
        asm!("{0} {1}", out(reg) _, inlateout(reg) b => _);
        asm!("", out("al") _, lateout("rcx") _);
        asm!("inst1\ninst2");
        asm!("inst1 {0}, 42\ninst2 {1}, 24", in(reg) a, out(reg) b);
        asm!("inst2 {1}, 24\ninst1 {0}, 42", in(reg) a, out(reg) b);
        asm!("inst1 {0}, 42\ninst2 {1}, 24", in(reg) a, out(reg) b);
        asm!("inst1\ninst2");
        asm!("inst1\ninst2");
        asm!("inst1\n\tinst2");
        asm!("inst1\ninst2\ninst3\ninst4");
    }
}
