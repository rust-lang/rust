//@ check-pass
//@ compile-flags: -Zrandomize-layout -Zlayout-seed=2464363

use std::cell::UnsafeCell;

#[derive(Default)]
struct A {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    e: UnsafeCell<u32>,
    f: UnsafeCell<u32>,
    g: UnsafeCell<u32>,
    h: UnsafeCell<u32>,
}

#[repr(transparent)]
struct B(A);

fn main() {
    let _b: &B = unsafe { std::mem::transmute(&A::default()) };
}
