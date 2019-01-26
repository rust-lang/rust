#![allow(stable_features)]

#![feature(extended_compare_and_swap)]
use std::sync::atomic::AtomicIsize;
use std::sync::atomic::Ordering::*;

static ATOMIC: AtomicIsize = AtomicIsize::new(0);

fn main() {
    // Make sure codegen can emit all the intrinsics correctly
    ATOMIC.compare_exchange(0, 1, Relaxed, Relaxed).ok();
    ATOMIC.compare_exchange(0, 1, Acquire, Relaxed).ok();
    ATOMIC.compare_exchange(0, 1, Release, Relaxed).ok();
    ATOMIC.compare_exchange(0, 1, AcqRel, Relaxed).ok();
    ATOMIC.compare_exchange(0, 1, SeqCst, Relaxed).ok();
    ATOMIC.compare_exchange(0, 1, Acquire, Acquire).ok();
    ATOMIC.compare_exchange(0, 1, AcqRel, Acquire).ok();
    ATOMIC.compare_exchange(0, 1, SeqCst, Acquire).ok();
    ATOMIC.compare_exchange(0, 1, SeqCst, SeqCst).ok();
    ATOMIC.compare_exchange_weak(0, 1, Relaxed, Relaxed).ok();
    ATOMIC.compare_exchange_weak(0, 1, Acquire, Relaxed).ok();
    ATOMIC.compare_exchange_weak(0, 1, Release, Relaxed).ok();
    ATOMIC.compare_exchange_weak(0, 1, AcqRel, Relaxed).ok();
    ATOMIC.compare_exchange_weak(0, 1, SeqCst, Relaxed).ok();
    ATOMIC.compare_exchange_weak(0, 1, Acquire, Acquire).ok();
    ATOMIC.compare_exchange_weak(0, 1, AcqRel, Acquire).ok();
    ATOMIC.compare_exchange_weak(0, 1, SeqCst, Acquire).ok();
    ATOMIC.compare_exchange_weak(0, 1, SeqCst, SeqCst).ok();
}
