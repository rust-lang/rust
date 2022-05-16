// ignore-windows: Concurrency on Windows is not supported yet.
#![feature(atomic_from_mut)]
#![feature(core_intrinsics)]

use std::intrinsics::atomic_load;
use std::sync::atomic::Ordering::*;
use std::sync::atomic::{AtomicU16, AtomicU32};

fn test_same_thread() {
    let mut qword = AtomicU32::new(42);
    assert_eq!(qword.load(Relaxed), 42);
    qword.store(0xabbafafa, Relaxed);

    let qword_mut = qword.get_mut();

    let dwords_mut = unsafe { std::mem::transmute::<&mut u32, &mut [u16; 2]>(qword_mut) };

    let (hi_mut, lo_mut) = dwords_mut.split_at_mut(1);

    let (hi, _) = (AtomicU16::from_mut(&mut hi_mut[0]), AtomicU16::from_mut(&mut lo_mut[0]));

    unsafe {
        //Equivalent to: hi.load(Ordering::SeqCst)
        atomic_load(hi.get_mut() as *mut u16); //~ ERROR: mixed-size access on an existing atomic object
    }
}

pub fn main() {
    test_same_thread();
}
