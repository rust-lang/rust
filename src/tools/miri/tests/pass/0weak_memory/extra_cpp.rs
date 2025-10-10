//@compile-flags: -Zmiri-ignore-leaks

// Tests operations not performable through C++'s atomic API
// but doable in safe (at least sound) Rust.

#![feature(atomic_from_mut)]

use std::sync::atomic::Ordering::*;
use std::sync::atomic::{AtomicU16, AtomicU32};
use std::thread::spawn;

fn static_atomic_mut(val: u32) -> &'static mut AtomicU32 {
    let ret = Box::leak(Box::new(AtomicU32::new(val)));
    ret
}

fn split_u32(dword: &mut u32) -> &mut [u16; 2] {
    unsafe { std::mem::transmute::<&mut u32, &mut [u16; 2]>(dword) }
}

fn mem_replace() {
    let mut x = AtomicU32::new(0);

    let old_x = std::mem::replace(&mut x, AtomicU32::new(42));

    assert_eq!(x.load(Relaxed), 42);
    assert_eq!(old_x.load(Relaxed), 0);
}

fn assign_to_mut() {
    let x = static_atomic_mut(0);
    x.store(1, Relaxed);

    *x = AtomicU32::new(2);

    assert_eq!(x.load(Relaxed), 2);
}

fn get_mut_write() {
    let x = static_atomic_mut(0);
    x.store(1, Relaxed);
    {
        let x_mut = x.get_mut();
        *x_mut = 2;
    }

    let j1 = spawn(move || x.load(Relaxed));

    let r1 = j1.join().unwrap();
    assert_eq!(r1, 2);
}

// This is technically doable in C++ with atomic_ref
// but little literature exists atm on its involvement
// in mixed size/atomicity accesses
fn from_mut_split() {
    let mut x: u32 = 0;

    {
        let x_atomic = AtomicU32::from_mut(&mut x);
        x_atomic.store(u32::from_be(0xabbafafa), Relaxed);
    }

    // Split the `AtomicU32` into two `AtomicU16`.
    // Crucially, there is no non-atomic access to `x`! All accesses are atomic, but of different size.
    let (x_hi, x_lo) = split_u32(&mut x).split_at_mut(1);

    let x_hi_atomic = AtomicU16::from_mut(&mut x_hi[0]);
    let x_lo_atomic = AtomicU16::from_mut(&mut x_lo[0]);

    assert_eq!(x_hi_atomic.load(Relaxed), u16::from_be(0xabba));
    assert_eq!(x_lo_atomic.load(Relaxed), u16::from_be(0xfafa));
}

pub fn main() {
    get_mut_write();
    from_mut_split();
    assign_to_mut();
    mem_replace();
}
