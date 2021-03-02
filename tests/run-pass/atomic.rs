use std::sync::atomic::{compiler_fence, fence, AtomicBool, AtomicIsize, AtomicU64, Ordering::*};

fn main() {
    atomic_bool();
    atomic_isize();
    atomic_u64();
    atomic_fences();
    weak_sometimes_fails();
}

fn atomic_bool() {
    static mut ATOMIC: AtomicBool = AtomicBool::new(false);

    unsafe {
        assert_eq!(*ATOMIC.get_mut(), false);
        ATOMIC.store(true, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), true);
        ATOMIC.fetch_or(false, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), true);
        ATOMIC.fetch_and(false, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), false);
        ATOMIC.fetch_nand(true, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), true);
        ATOMIC.fetch_xor(true, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), false);
    }
}
// There isn't a trait to use to make this generic, so just use a macro
macro_rules! compare_exchange_weak_loop {
    ($atom:expr, $from:expr, $to:expr, $succ_order:expr, $fail_order:expr) => {
        loop {
            match $atom.compare_exchange_weak($from, $to, $succ_order, $fail_order) {
                Ok(n) => {
                    assert_eq!(n, $from);
                    break;
                }
                Err(n) => assert_eq!(n, $from),
            }
        }
    };
}
fn atomic_isize() {
    static ATOMIC: AtomicIsize = AtomicIsize::new(0);

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
    compare_exchange_weak_loop!(ATOMIC, 0, 1, Relaxed, Relaxed);
    assert_eq!(ATOMIC.compare_exchange_weak(0, 2, Acquire, Relaxed), Err(1));
    assert_eq!(ATOMIC.compare_exchange_weak(0, 1, Release, Relaxed), Err(1));
    compare_exchange_weak_loop!(ATOMIC, 1, 0, AcqRel, Relaxed);
    assert_eq!(ATOMIC.load(Relaxed), 0);
    ATOMIC.compare_exchange_weak(0, 1, AcqRel, Relaxed).ok();
    ATOMIC.compare_exchange_weak(0, 1, SeqCst, Relaxed).ok();
    ATOMIC.compare_exchange_weak(0, 1, Acquire, Acquire).ok();
    ATOMIC.compare_exchange_weak(0, 1, AcqRel, Acquire).ok();
    ATOMIC.compare_exchange_weak(0, 1, SeqCst, Acquire).ok();
    ATOMIC.compare_exchange_weak(0, 1, SeqCst, SeqCst).ok();
}

fn atomic_u64() {
    static ATOMIC: AtomicU64 = AtomicU64::new(0);

    ATOMIC.store(1, SeqCst);
    assert_eq!(ATOMIC.compare_exchange(0, 0x100, AcqRel, Acquire), Err(1));
    compare_exchange_weak_loop!(ATOMIC, 1, 0x100, AcqRel, Acquire);
    assert_eq!(ATOMIC.load(Relaxed), 0x100);

    assert_eq!(ATOMIC.fetch_max(0x10, SeqCst), 0x100);
    assert_eq!(ATOMIC.fetch_max(0x100, SeqCst), 0x100);
    assert_eq!(ATOMIC.fetch_max(0x1000, SeqCst), 0x100);
    assert_eq!(ATOMIC.fetch_max(0x1000, SeqCst), 0x1000);
    assert_eq!(ATOMIC.fetch_max(0x2000, SeqCst), 0x1000);
    assert_eq!(ATOMIC.fetch_max(0x2000, SeqCst), 0x2000);

    assert_eq!(ATOMIC.fetch_min(0x2000, SeqCst), 0x2000);
    assert_eq!(ATOMIC.fetch_min(0x2000, SeqCst), 0x2000);
    assert_eq!(ATOMIC.fetch_min(0x1000, SeqCst), 0x2000);
    assert_eq!(ATOMIC.fetch_min(0x1000, SeqCst), 0x1000);
    assert_eq!(ATOMIC.fetch_min(0x100, SeqCst), 0x1000);
    assert_eq!(ATOMIC.fetch_min(0x10, SeqCst), 0x100);
}

fn atomic_fences() {
    fence(SeqCst);
    fence(Release);
    fence(Acquire);
    fence(AcqRel);
    compiler_fence(SeqCst);
    compiler_fence(Release);
    compiler_fence(Acquire);
    compiler_fence(AcqRel);
}

fn weak_sometimes_fails() {
    let atomic = AtomicBool::new(false);
    let tries = 100;
    for _ in 0..tries {
        let cur = atomic.load(Relaxed);
        // Try (weakly) to flip the flag.
        if atomic.compare_exchange_weak(cur, !cur, Relaxed, Relaxed).is_err() {
            // We failed, so return and skip the panic.
            return;
        }
    }
    panic!("compare_exchange_weak succeeded {} tries in a row", tries);
}
