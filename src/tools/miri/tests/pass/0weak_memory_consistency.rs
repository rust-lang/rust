//@compile-flags: -Zmiri-ignore-leaks -Zmiri-disable-stacked-borrows

// The following tests check whether our weak memory emulation produces
// any inconsistent execution outcomes
//
// Due to the random nature of choosing valid stores, it is always
// possible that our tests spuriously succeeds: even though our weak
// memory emulation code has incorrectly identified a store in
// modification order as being valid, it may be never chosen by
// the RNG and never observed in our tests.
//
// To mitigate this, each test is ran enough times such that the chance
// of spurious success is very low. These tests never spuriously fail.

// Test cases and their consistent outcomes are from
// http://svr-pes20-cppmem.cl.cam.ac.uk/cppmem/
// Based on
// M. Batty, S. Owens, S. Sarkar, P. Sewell and T. Weber,
// "Mathematizing C++ concurrency", ACM SIGPLAN Notices, vol. 46, no. 1, pp. 55-66, 2011.
// Available: https://ss265.host.cs.st-andrews.ac.uk/papers/n3132.pdf.

use std::sync::atomic::Ordering::*;
use std::sync::atomic::{fence, AtomicBool, AtomicI32};
use std::thread::spawn;

#[derive(Copy, Clone)]
struct EvilSend<T>(pub T);

unsafe impl<T> Send for EvilSend<T> {}
unsafe impl<T> Sync for EvilSend<T> {}

// We can't create static items because we need to run each test
// multiple times
fn static_atomic(val: i32) -> &'static AtomicI32 {
    let ret = Box::leak(Box::new(AtomicI32::new(val)));
    ret.store(val, Relaxed); // work around https://github.com/rust-lang/miri/issues/2164
    ret
}
fn static_atomic_bool(val: bool) -> &'static AtomicBool {
    let ret = Box::leak(Box::new(AtomicBool::new(val)));
    ret.store(val, Relaxed); // work around https://github.com/rust-lang/miri/issues/2164
    ret
}

// Spins until it acquires a pre-determined value.
fn acquires_value(loc: &AtomicI32, val: i32) -> i32 {
    while loc.load(Acquire) != val {
        std::hint::spin_loop();
    }
    val
}

fn test_corr() {
    let x = static_atomic(0);
    let y = static_atomic(0);

    let j1 = spawn(move || {
        x.store(1, Relaxed);
        x.store(2, Relaxed);
    });

    #[rustfmt::skip]
    let j2 = spawn(move || {
        let r2 = x.load(Relaxed); // -------------------------------------+
        y.store(1, Release); // ---------------------+                    |
        r2 //                                        |                    |
    }); //                                           |                    |
    #[rustfmt::skip] //                              |synchronizes-with   |happens-before
    let j3 = spawn(move || { //                      |                    |
        acquires_value(&y, 1); // <------------------+                    |
        x.load(Relaxed) // <----------------------------------------------+
        // The two reads on x are ordered by hb, so they cannot observe values
        // differently from the modification order. If the first read observed
        // 2, then the second read must observe 2 as well.
    });

    j1.join().unwrap();
    let r2 = j2.join().unwrap();
    let r3 = j3.join().unwrap();
    if r2 == 2 {
        assert_eq!(r3, 2);
    }
}

fn test_wrc() {
    let x = static_atomic(0);
    let y = static_atomic(0);

    #[rustfmt::skip]
    let j1 = spawn(move || {
        x.store(1, Release); // ---------------------+---------------------+
    }); //                                           |                     |
    #[rustfmt::skip] //                              |synchronizes-with    |
    let j2 = spawn(move || { //                      |                     |
        acquires_value(&x, 1); // <------------------+                     |
        y.store(1, Release); // ---------------------+                     |happens-before
    }); //                                           |                     |
    #[rustfmt::skip] //                              |synchronizes-with    |
    let j3 = spawn(move || { //                      |                     |
        acquires_value(&y, 1); // <------------------+                     |
        x.load(Relaxed) // <-----------------------------------------------+
    });

    j1.join().unwrap();
    j2.join().unwrap();
    let r3 = j3.join().unwrap();

    assert_eq!(r3, 1);
}

fn test_message_passing() {
    let mut var = 0u32;
    let ptr = &mut var as *mut u32;
    let x = EvilSend(ptr);
    let y = static_atomic(0);

    #[rustfmt::skip]
    let j1 = spawn(move || {
        unsafe { *x.0 = 1 }; // -----------------------------------------+
        y.store(1, Release); // ---------------------+                   |
    }); //                                           |                   |
    #[rustfmt::skip] //                              |synchronizes-with  | happens-before
    let j2 = spawn(move || { //                      |                   |
        acquires_value(&y, 1); // <------------------+                   |
        unsafe { *x.0 } // <---------------------------------------------+
    });

    j1.join().unwrap();
    let r2 = j2.join().unwrap();

    assert_eq!(r2, 1);
}

// LB+acq_rel+acq_rel
fn test_load_buffering_acq_rel() {
    let x = static_atomic(0);
    let y = static_atomic(0);
    let j1 = spawn(move || {
        let r1 = x.load(Acquire);
        y.store(1, Release);
        r1
    });

    let j2 = spawn(move || {
        let r2 = y.load(Acquire);
        x.store(1, Release);
        r2
    });

    let r1 = j1.join().unwrap();
    let r2 = j2.join().unwrap();

    // 3 consistent outcomes: (0,0), (0,1), (1,0)
    assert_ne!((r1, r2), (1, 1));
}

fn test_mixed_access() {
    /*
    int main() {
      atomic_int x = 0;
      {{{
        x.store(1, mo_relaxed);
      }}}

      x.store(2, mo_relaxed);

      {{{
        r1 = x.load(mo_relaxed);
      }}}

      return 0;
    }
        */
    let x = static_atomic(0);

    spawn(move || {
        x.store(1, Relaxed);
    })
    .join()
    .unwrap();

    x.store(2, Relaxed);

    let r2 = spawn(move || x.load(Relaxed)).join().unwrap();

    assert_eq!(r2, 2);
}

// The following two tests are taken from Repairing Sequential Consistency in C/C++11
// by Lahav et al.
// https://plv.mpi-sws.org/scfix/paper.pdf

// Test case SB
fn test_sc_store_buffering() {
    let x = static_atomic(0);
    let y = static_atomic(0);

    let j1 = spawn(move || {
        x.store(1, SeqCst);
        y.load(SeqCst)
    });

    let j2 = spawn(move || {
        y.store(1, SeqCst);
        x.load(SeqCst)
    });

    let a = j1.join().unwrap();
    let b = j2.join().unwrap();

    assert_ne!((a, b), (0, 0));
}

fn test_single_thread() {
    let x = AtomicI32::new(42);

    assert_eq!(x.load(Relaxed), 42);

    x.store(43, Relaxed);

    assert_eq!(x.load(Relaxed), 43);
}

fn test_sync_through_rmw_and_fences() {
    // Example from https://github.com/llvm/llvm-project/issues/56450#issuecomment-1183695905
    #[no_mangle]
    pub fn rdmw(storing: &AtomicI32, sync: &AtomicI32, loading: &AtomicI32) -> i32 {
        storing.store(1, Relaxed);
        fence(Release);
        sync.fetch_add(0, Relaxed);
        fence(Acquire);
        loading.load(Relaxed)
    }

    let x = static_atomic(0);
    let y = static_atomic(0);
    let z = static_atomic(0);

    // Since each thread is so short, we need to make sure that they truely run at the same time
    // Otherwise t1 will finish before t2 even starts
    let go = static_atomic_bool(false);

    let t1 = spawn(move || {
        while !go.load(Relaxed) {}
        rdmw(y, x, z)
    });

    let t2 = spawn(move || {
        while !go.load(Relaxed) {}
        rdmw(z, x, y)
    });

    go.store(true, Relaxed);

    let a = t1.join().unwrap();
    let b = t2.join().unwrap();
    assert_ne!((a, b), (0, 0));
}

// Test case by @SabrinaJewson
// https://github.com/rust-lang/miri/issues/2301#issuecomment-1221502757
// Demonstrating C++20 SC access changes
fn test_iriw_sc_rlx() {
    let x = static_atomic_bool(false);
    let y = static_atomic_bool(false);

    x.store(false, Relaxed);
    y.store(false, Relaxed);

    let a = spawn(move || x.store(true, Relaxed));
    let b = spawn(move || y.store(true, Relaxed));
    let c = spawn(move || {
        while !x.load(SeqCst) {}
        y.load(SeqCst)
    });
    let d = spawn(move || {
        while !y.load(SeqCst) {}
        x.load(SeqCst)
    });

    a.join().unwrap();
    b.join().unwrap();
    let c = c.join().unwrap();
    let d = d.join().unwrap();

    assert!(c || d);
}

pub fn main() {
    for _ in 0..50 {
        test_single_thread();
        test_mixed_access();
        test_load_buffering_acq_rel();
        test_message_passing();
        test_wrc();
        test_corr();
        test_sc_store_buffering();
        test_sync_through_rmw_and_fences();
        test_iriw_sc_rlx();
    }
}
