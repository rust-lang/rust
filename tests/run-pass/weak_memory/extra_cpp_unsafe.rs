// ignore-windows: Concurrency on Windows is not supported yet.
// compile-flags: -Zmiri-ignore-leaks

// Tests operations not perfomable through C++'s atomic API
// but doable in unsafe Rust which we think *should* be fine.
// Nonetheless they may be determined as inconsistent with the
// memory model in the future.

#![feature(atomic_from_mut)]
#![feature(core_intrinsics)]

use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::*;
use std::thread::spawn;

fn static_atomic(val: u32) -> &'static AtomicU32 {
    let ret = Box::leak(Box::new(AtomicU32::new(val)));
    ret
}

fn split_u32_ptr(dword: *const u32) -> *const [u16; 2] {
    unsafe { std::mem::transmute::<*const u32, *const [u16; 2]>(dword) }
}

// We allow non-atomic and atomic reads to race
fn racing_mixed_atomicity_read() {
    let x = static_atomic(0);
    x.store(42, Relaxed);

    let j1 = spawn(move || x.load(Relaxed));

    let j2 = spawn(move || {
        let x_ptr = x as *const AtomicU32 as *const u32;
        unsafe { std::intrinsics::atomic_load_relaxed(x_ptr) }
    });

    let r1 = j1.join().unwrap();
    let r2 = j2.join().unwrap();

    assert_eq!(r1, 42);
    assert_eq!(r2, 42);
}

// We allow mixed-size atomic reads to race
fn racing_mixed_size_read() {
    let x = static_atomic(0);

    let j1 = spawn(move || {
        x.load(Relaxed);
    });

    let j2 = spawn(move || {
        let x_ptr = x as *const AtomicU32 as *const u32;
        let x_split = split_u32_ptr(x_ptr);
        unsafe {
            let hi = &(*x_split)[0] as *const u16;
            std::intrinsics::atomic_load_relaxed(hi);
        }
    });

    j1.join().unwrap();
    j2.join().unwrap();
}

// And we allow the combination of both of the above.
fn racing_mixed_atomicity_and_size_read() {
    let x = static_atomic(u32::from_be(0xabbafafa));

    let j1 = spawn(move || {
        x.load(Relaxed);
    });

    let j2 = spawn(move || {
        let x_ptr = x as *const AtomicU32 as *const u32;
        unsafe { *x_ptr };
    });

    let j3 = spawn(move || {
        let x_ptr = x as *const AtomicU32 as *const u32;
        let x_split = split_u32_ptr(x_ptr);
        unsafe {
            let hi = &(*x_split)[0] as *const u16;
            std::intrinsics::atomic_load_relaxed(hi)
        }
    });

    j1.join().unwrap();
    j2.join().unwrap();
    let r3 = j3.join().unwrap();

    assert_eq!(r3, u16::from_be(0xabba));
}

pub fn main() {
    racing_mixed_atomicity_read();
    racing_mixed_size_read();
    racing_mixed_atomicity_and_size_read();
}
