use std::sync::atomic::{AtomicIsize, Ordering::*};

static ATOMIC: AtomicIsize = AtomicIsize::new(0);

fn main() {
    // Make sure trans can emit all the intrinsics correctly
    assert_eq!(ATOMIC.compare_exchange(0, 1, Relaxed, Relaxed), Ok(0));
    assert_eq!(ATOMIC.compare_exchange(0, 2, Acquire, Relaxed), Err(1));
    assert_eq!(ATOMIC.compare_exchange(0, 1, Release, Relaxed), Err(1));
    assert_eq!(ATOMIC.compare_exchange(1, 0, AcqRel, Relaxed), Ok(1));
    ATOMIC.compare_exchange(0, 1, SeqCst, Relaxed).ok();
    ATOMIC.compare_exchange(0, 1, Acquire, Acquire).ok();
    ATOMIC.compare_exchange(0, 1, AcqRel, Acquire).ok();
    ATOMIC.compare_exchange(0, 1, SeqCst, Acquire).ok();
    ATOMIC.compare_exchange(0, 1, SeqCst, SeqCst).ok();

    ATOMIC.store(0, SeqCst);

    assert_eq!(ATOMIC.compare_exchange_weak(0, 1, Relaxed, Relaxed), Ok(0));
    assert_eq!(ATOMIC.compare_exchange_weak(0, 2, Acquire, Relaxed), Err(1));
    assert_eq!(ATOMIC.compare_exchange_weak(0, 1, Release, Relaxed), Err(1));
    assert_eq!(ATOMIC.compare_exchange_weak(1, 0, AcqRel, Relaxed), Ok(1));
    ATOMIC.compare_exchange_weak(0, 1, AcqRel, Relaxed).ok();
    ATOMIC.compare_exchange_weak(0, 1, SeqCst, Relaxed).ok();
    ATOMIC.compare_exchange_weak(0, 1, Acquire, Acquire).ok();
    ATOMIC.compare_exchange_weak(0, 1, AcqRel, Acquire).ok();
    ATOMIC.compare_exchange_weak(0, 1, SeqCst, Acquire).ok();
    ATOMIC.compare_exchange_weak(0, 1, SeqCst, SeqCst).ok();
}
