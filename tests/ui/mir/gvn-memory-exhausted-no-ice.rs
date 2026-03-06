//@ compile-flags: -Zmir-opt-level=5
//@ check-pass
//@ only-64bit

#![feature(maybe_uninit_as_bytes)]

use std::mem::MaybeUninit;
use std::ptr::from_ref;

const N: usize = 0x0000_7ff0_0000_0000;

#[inline(never)]
pub fn g(n: &u8) {
    let mut xs = MaybeUninit::<[u8; N]>::uninit();
    let base = from_ref(&xs.as_bytes()[0]).addr();
    let index = from_ref(n).addr() - base;
    xs.as_bytes_mut()[index].write(42);
}

pub fn main() {
    let n = Box::new(27);
    g(&n);
    std::process::exit(*n as i32);
}
