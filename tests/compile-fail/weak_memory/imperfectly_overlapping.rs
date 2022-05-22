// ignore-windows: Concurrency on Windows is not supported yet.
#![feature(atomic_from_mut)]
#![feature(core_intrinsics)]

use std::intrinsics::atomic_load;
use std::sync::atomic::Ordering::*;
use std::sync::atomic::{AtomicU16, AtomicU32};

fn split_u32(dword: &mut u32) -> &mut [u16; 2] {
    unsafe { std::mem::transmute::<&mut u32, &mut [u16; 2]>(dword) }
}

fn test_same_thread() {
    let mut dword = AtomicU32::new(42);
    assert_eq!(dword.load(Relaxed), 42);
    dword.store(0xabbafafa, Relaxed);

    let dword_mut = dword.get_mut();

    let words_mut = split_u32(dword_mut);

    let (hi_mut, lo_mut) = words_mut.split_at_mut(1);

    let (hi, _) = (AtomicU16::from_mut(&mut hi_mut[0]), AtomicU16::from_mut(&mut lo_mut[0]));

    unsafe {
        // Equivalent to: hi.load(Ordering::SeqCst)
        // We need to use intrisics to for precise error location
        atomic_load(hi.get_mut() as *mut u16); //~ ERROR: mixed-size access on an existing atomic object
    }
}

pub fn main() {
    test_same_thread();
}
