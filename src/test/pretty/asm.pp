#![feature(prelude_import)]
#![no_std]
#![feature(asm)]
#[prelude_import]
use ::std::prelude::v1::*;
#[macro_use]
extern crate std;

// pretty-mode:expanded
// pp-exact:asm.pp
// only-x86_64

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
        asm!("", out("al") _, lateout("rbx") _);
    }
}
