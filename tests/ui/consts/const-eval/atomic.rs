//@ check-pass
#![feature(const_atomic, const_raw_ptr_comparison)]

use std::sync::atomic::*;
use std::sync::atomic::Ordering::*;

const BOOL: () = {
    let mut atomic: AtomicBool = AtomicBool::new(false);

    unsafe {
        assert!(*atomic.get_mut() == false);
        atomic.store(true, SeqCst);
        assert!(*atomic.get_mut() == true);
        atomic.fetch_or(false, SeqCst);
        assert!(*atomic.get_mut() == true);
        atomic.fetch_and(false, SeqCst);
        assert!(*atomic.get_mut() == false);
        atomic.fetch_nand(true, SeqCst);
        assert!(*atomic.get_mut() == true);
        atomic.fetch_xor(true, SeqCst);
        assert!(*atomic.get_mut() == false);
    }
};

const FENCES: () = {
    fence(SeqCst);
    fence(Release);
    fence(Acquire);
    fence(AcqRel);
    compiler_fence(SeqCst);
    compiler_fence(Release);
    compiler_fence(Acquire);
    compiler_fence(AcqRel);
};

const fn compare_result_u64(a: Result<u64, u64>, b: Result<u64, u64>) -> bool {
    match (a, b) {
        (Ok(a), Ok(b)) => a == b,
        (Err(a), Err(b)) => a == b,
        _ => false,
    }
}

#[cfg(target_has_atomic = "64")]
const ATOMIC_INT: () = {
    let atomic = AtomicU64::new(0);

    atomic.store(1, SeqCst);
    assert!(compare_result_u64(atomic.compare_exchange(0, 0x100, AcqRel, Acquire), Err(1)));
    assert!(compare_result_u64(atomic.compare_exchange(0, 1, Release, Relaxed), Err(1)));
    assert!(compare_result_u64(atomic.compare_exchange(1, 0, AcqRel, Relaxed), Ok(1)));
    assert!(compare_result_u64(atomic.compare_exchange(0, 1, Relaxed, Relaxed), Ok(0)));
    // compare_exchange_weak always succeeds when possible, but that is not a guarantee.
    assert!(compare_result_u64(atomic.compare_exchange_weak(1, 0x100, AcqRel, Acquire), Ok(1)));
    assert!(compare_result_u64(atomic.compare_exchange_weak(0, 2, Acquire, Relaxed), Err(0x100)));
    assert!(compare_result_u64(atomic.compare_exchange_weak(0, 1, Release, Relaxed), Err(0x100)));
    assert!(atomic.load(Relaxed) == 0x100);

    assert!(atomic.fetch_max(0x10, SeqCst) == 0x100);
    assert!(atomic.fetch_max(0x100, SeqCst) == 0x100);
    assert!(atomic.fetch_max(0x1000, SeqCst) == 0x100);
    assert!(atomic.fetch_max(0x1000, SeqCst) == 0x1000);
    assert!(atomic.fetch_max(0x2000, SeqCst) == 0x1000);
    assert!(atomic.fetch_max(0x2000, SeqCst) == 0x2000);

    assert!(atomic.fetch_min(0x2000, SeqCst) == 0x2000);
    assert!(atomic.fetch_min(0x2000, SeqCst) == 0x2000);
    assert!(atomic.fetch_min(0x1000, SeqCst) == 0x2000);
    assert!(atomic.fetch_min(0x1000, SeqCst) == 0x1000);
    assert!(atomic.fetch_min(0x100, SeqCst) == 0x1000);
    assert!(atomic.fetch_min(0x10, SeqCst) == 0x100);

    assert!(atomic.swap(1, SeqCst) == 0x10);
    assert!(atomic.load(Relaxed) == 1);

    let atomic_signed = AtomicI64::new(0);
    assert!(atomic_signed.fetch_min(-1, SeqCst) == 0);
    assert!(atomic_signed.load(SeqCst) == -1);
    assert!(atomic_signed.fetch_min(1, SeqCst) == -1);
    assert!(atomic_signed.load(SeqCst) == -1);
    assert!(atomic_signed.fetch_max(1, SeqCst) == -1);
    assert!(atomic_signed.load(SeqCst) == 1);
    assert!(atomic_signed.fetch_max(-1, SeqCst) == 1);
    assert!(atomic_signed.load(SeqCst) == 1);
};

const ATOMIC_PTR: () = {
    use std::ptr;
    let array = [0i32; 4]; // a target to point to, to test provenance things
    let x = array.as_ptr() as *mut i32;

    let ptr = AtomicPtr::<i32>::new(ptr::null_mut());
    assert!(ptr.load(Relaxed).guaranteed_eq(0 as *mut i32).unwrap());
    ptr.store(ptr::without_provenance_mut(13), SeqCst);
    assert!(ptr.swap(x, Relaxed).guaranteed_eq(13 as *mut i32).unwrap());
    unsafe { assert!(ptr.load(Acquire).read() == 0) };
};

fn main() {}
