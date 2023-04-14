//@compile-flags: -Zmiri-ignore-leaks

// Tests operations not performable through C++'s atomic API
// but doable in unsafe Rust which we think *should* be fine.
// Nonetheless they may be determined as inconsistent with the
// memory model in the future.

#![feature(atomic_from_mut)]

use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::*;
use std::thread::spawn;

fn static_atomic(val: u32) -> &'static AtomicU32 {
    let ret = Box::leak(Box::new(AtomicU32::new(val)));
    ret
}

// We allow perfectly overlapping non-atomic and atomic reads to race
fn racing_mixed_atomicity_read() {
    let x = static_atomic(0);
    x.store(42, Relaxed);

    let j1 = spawn(move || x.load(Relaxed));

    let j2 = spawn(move || {
        let x_ptr = x as *const AtomicU32 as *const u32;
        unsafe { x_ptr.read() }
    });

    let r1 = j1.join().unwrap();
    let r2 = j2.join().unwrap();

    assert_eq!(r1, 42);
    assert_eq!(r2, 42);
}

pub fn main() {
    racing_mixed_atomicity_read();
}
