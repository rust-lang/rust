// ignore-windows: Concurrency on Windows is not supported yet.
#![feature(atomic_from_mut)]

use std::sync::atomic::Ordering::*;
use std::sync::atomic::{AtomicU16, AtomicU32};

// Strictly speaking, atomic accesses that imperfectly overlap with existing
// atomic objects are UB. Nonetheless we'd like to provide a sane value when
// the access is not racy.
fn test_same_thread() {
    let mut qword = AtomicU32::new(42);
    assert_eq!(qword.load(Relaxed), 42);
    qword.store(u32::to_be(0xabbafafa), Relaxed);

    let qword_mut = qword.get_mut();

    let dwords_mut = unsafe { std::mem::transmute::<&mut u32, &mut [u16; 2]>(qword_mut) };

    let (hi_mut, lo_mut) = dwords_mut.split_at_mut(1);

    let (hi, lo) = (AtomicU16::from_mut(&mut hi_mut[0]), AtomicU16::from_mut(&mut lo_mut[0]));

    assert_eq!(u16::from_be(hi.load(Relaxed)), 0xabba);
    assert_eq!(u16::from_be(lo.load(Relaxed)), 0xfafa);
}

pub fn main() {
    test_same_thread();
}
