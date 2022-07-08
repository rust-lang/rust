//@ignore-windows: Concurrency on Windows is not supported yet.
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
// of spurious success is very low. These tests never supriously fail.

// Test cases and their consistent outcomes are from
// http://svr-pes20-cppmem.cl.cam.ac.uk/cppmem/
// Based on
// M. Batty, S. Owens, S. Sarkar, P. Sewell and T. Weber,
// "Mathematizing C++ concurrency", ACM SIGPLAN Notices, vol. 46, no. 1, pp. 55-66, 2011.
// Available: https://ss265.host.cs.st-andrews.ac.uk/papers/n3132.pdf.

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::*;
use std::thread::spawn;

#[derive(Copy, Clone)]
struct EvilSend<T>(pub T);

unsafe impl<T> Send for EvilSend<T> {}
unsafe impl<T> Sync for EvilSend<T> {}

// We can't create static items because we need to run each test
// multiple times
fn static_atomic(val: usize) -> &'static AtomicUsize {
    let ret = Box::leak(Box::new(AtomicUsize::new(val)));
    ret
}

// Spins until it acquires a pre-determined value.
fn acquires_value(loc: &AtomicUsize, val: usize) -> usize {
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
    let x = AtomicUsize::new(42);

    assert_eq!(x.load(Relaxed), 42);

    x.store(43, Relaxed);

    assert_eq!(x.load(Relaxed), 43);
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
    }
}
