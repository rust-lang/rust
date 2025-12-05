//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
//@compile-flags: -Zmiri-strict-provenance

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

use std::sync::atomic::Ordering::*;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicPtr, AtomicUsize, compiler_fence, fence};

fn main() {
    atomic_bool();
    atomic_all_ops();
    atomic_fences();
    atomic_ptr();
    weak_sometimes_fails();

    #[cfg(target_has_atomic = "64")]
    atomic_u64();
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

/// Make sure we can handle all the intrinsics
fn atomic_all_ops() {
    static ATOMIC: AtomicIsize = AtomicIsize::new(0);
    static ATOMIC_UNSIGNED: AtomicUsize = AtomicUsize::new(0);

    let load_orders = [Relaxed, Acquire, SeqCst];
    let stored_orders = [Relaxed, Release, SeqCst];
    let rmw_orders = [Relaxed, Release, Acquire, AcqRel, SeqCst];

    // loads
    for o in load_orders {
        ATOMIC.load(o);
    }

    // stores
    for o in stored_orders {
        ATOMIC.store(1, o);
    }

    // most RMWs
    for o in rmw_orders {
        ATOMIC.swap(0, o);
        ATOMIC.fetch_or(0, o);
        ATOMIC.fetch_xor(0, o);
        ATOMIC.fetch_and(0, o);
        ATOMIC.fetch_nand(0, o);
        ATOMIC.fetch_add(0, o);
        ATOMIC.fetch_sub(0, o);
        ATOMIC.fetch_min(0, o);
        ATOMIC.fetch_max(0, o);
        ATOMIC_UNSIGNED.fetch_min(0, o);
        ATOMIC_UNSIGNED.fetch_max(0, o);
    }

    // RMWs with separate failure ordering
    for o1 in rmw_orders {
        for o2 in load_orders {
            let _res = ATOMIC.compare_exchange(0, 0, o1, o2);
            let _res = ATOMIC.compare_exchange_weak(0, 0, o1, o2);
        }
    }
}

#[cfg(target_has_atomic = "64")]
fn atomic_u64() {
    use std::sync::atomic::AtomicU64;
    static ATOMIC: AtomicU64 = AtomicU64::new(0);

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

    ATOMIC.store(1, SeqCst);
    assert_eq!(ATOMIC.compare_exchange(0, 0x100, AcqRel, Acquire), Err(1));
    assert_eq!(ATOMIC.compare_exchange(0, 1, Release, Relaxed), Err(1));
    assert_eq!(ATOMIC.compare_exchange(1, 0, AcqRel, Relaxed), Ok(1));
    assert_eq!(ATOMIC.compare_exchange(0, 1, Relaxed, Relaxed), Ok(0));
    compare_exchange_weak_loop!(ATOMIC, 1, 0x100, AcqRel, Acquire);
    assert_eq!(ATOMIC.compare_exchange_weak(0, 2, Acquire, Relaxed), Err(0x100));
    assert_eq!(ATOMIC.compare_exchange_weak(0, 1, Release, Relaxed), Err(0x100));
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

fn atomic_ptr() {
    use std::ptr;
    let array: Vec<i32> = (0..100).into_iter().collect(); // a target to point to, to test provenance things
    let x = array.as_ptr() as *mut i32;

    let ptr = AtomicPtr::<i32>::new(ptr::null_mut());
    assert!(ptr.load(Relaxed).addr() == 0);
    ptr.store(ptr::without_provenance_mut(13), SeqCst);
    assert!(ptr.swap(x, Relaxed).addr() == 13);
    unsafe { assert!(*ptr.load(Acquire) == 0) };

    // comparison ignores provenance
    assert_eq!(
        ptr.compare_exchange(
            (&mut 0 as *mut i32).with_addr(x.addr()),
            ptr::without_provenance_mut(0),
            SeqCst,
            SeqCst
        )
        .unwrap()
        .addr(),
        x.addr(),
    );
    assert_eq!(
        ptr.compare_exchange(
            (&mut 0 as *mut i32).with_addr(x.addr()),
            ptr::without_provenance_mut(0),
            SeqCst,
            SeqCst
        )
        .unwrap_err()
        .addr(),
        0,
    );
    ptr.store(x, Relaxed);

    assert_eq!(ptr.fetch_ptr_add(13, AcqRel).addr(), x.addr());
    unsafe { assert_eq!(*ptr.load(SeqCst), 13) }; // points to index 13 now
    assert_eq!(ptr.fetch_ptr_sub(4, AcqRel).addr(), x.addr() + 13 * 4);
    unsafe { assert_eq!(*ptr.load(SeqCst), 9) };
    assert_eq!(ptr.fetch_or(3, AcqRel).addr(), x.addr() + 9 * 4); // ptr is 4-aligned, so set the last 2 bits
    assert_eq!(ptr.fetch_and(!3, AcqRel).addr(), (x.addr() + 9 * 4) | 3); // and unset them again
    unsafe { assert_eq!(*ptr.load(SeqCst), 9) };
    assert_eq!(ptr.fetch_xor(0xdeadbeef, AcqRel).addr(), x.addr() + 9 * 4);
    assert_eq!(ptr.fetch_xor(0xdeadbeef, AcqRel).addr(), (x.addr() + 9 * 4) ^ 0xdeadbeef);
    unsafe { assert_eq!(*ptr.load(SeqCst), 9) }; // after XORing twice with the same thing, we get our ptr back
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
