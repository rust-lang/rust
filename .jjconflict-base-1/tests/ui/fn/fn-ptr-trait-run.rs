#![feature(fn_static)]
//@ run-pass

use std::ops::FnPtr;

fn bar(a: u64) -> u64 {
    a
}

fn main() {
    type F = fn(u64) -> u64;
    let b: F = bar;
    assert_eq!(b.addr(), bar as *const () as usize);
    assert_eq!(b(42), unsafe { F::from_ptr(b.as_ptr())(42) });
}
