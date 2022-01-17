// ignore-windows: Concurrency on Windows is not supported yet.
// compile-flags: -Zmiri-ignore-leaks -Zmiri-disable-stacked-borrows

// Weak memory emulation tests. All of the following test if
// our weak memory emulation produces any inconsistent execution outcomes
//
// Due to the random nature of choosing valid stores, it is always
// possible that our tests spuriously succeeds: even though our weak
// memory emulation code has incorrectly identified a store in
// modification order as being valid, it may be never chosen by
// the RNG and never observed in our tests.
//
// To mitigate this, each test is ran enough times such that the chance
// of spurious success is very low. These tests never supriously fail.
//
// Note that we can't effectively test whether our weak memory emulation
// can produce *all* consistent execution outcomes. This may be possible
// if Miri's scheduler is sufficiently random and explores all possible
// interleavings of our small test cases after a reasonable number of runs.
// However, since Miri's scheduler is not even pre-emptive, there will
// always be possible interleavings (and possible execution outcomes),
// that can never be observed regardless of how weak memory emulation is
// implemented.

// Test cases and their consistent outcomes are from
// http://svr-pes20-cppmem.cl.cam.ac.uk/cppmem/
// Based on
// M. Batty, S. Owens, S. Sarkar, P. Sewell and T. Weber,
// "Mathematizing C++ concurrency", ACM SIGPLAN Notices, vol. 46, no. 1, pp. 55-66, 2011.
// Available: https://ss265.host.cs.st-andrews.ac.uk/papers/n3132.pdf.

use std::sync::atomic::Ordering::*;
use std::sync::atomic::{fence, AtomicUsize};
use std::thread::{spawn, yield_now};

#[derive(Copy, Clone)]
struct EvilSend<T>(pub T);

unsafe impl<T> Send for EvilSend<T> {}
unsafe impl<T> Sync for EvilSend<T> {}

// We can't create static items because we need to run each test
// multiple tests
fn static_atomic(val: usize) -> &'static AtomicUsize {
    let ret = Box::leak(Box::new(AtomicUsize::new(val)));
    // A workaround to put the initialisation value in the store buffer
    ret.store(val, Relaxed);
    ret
}

// Spins and yields until until acquires a pre-determined value
fn acquires_value(loc: &AtomicUsize, val: usize) -> usize {
    while loc.load(Acquire) != val {
        yield_now();
    }
    val
}

fn reads_value(loc: &AtomicUsize, val: usize) -> usize {
    while loc.load(Relaxed) != val {
        yield_now();
    }
    val
}

// https://plv.mpi-sws.org/scfix/paper.pdf
// 2.2 Second Problem: SC Fences are Too Weak
fn test_rwc_syncs() {
    /*
    int main() {
        atomic_int x = 0;
        atomic_int y = 0;

        {{{ x.store(1,mo_relaxed);
        ||| { r1=x.load(mo_relaxed).readsvalue(1);
              fence(mo_seq_cst);
              r2=y.load(mo_relaxed); }
        ||| { y.store(1,mo_relaxed);
              fence(mo_seq_cst);
              r3=x.load(mo_relaxed); }
        }}}
        return 0;
    }
    */
    let x = static_atomic(0);
    let y = static_atomic(0);

    let j1 = spawn(move || {
        x.store(1, Relaxed);
    });

    let j2 = spawn(move || {
        reads_value(&x, 1);
        fence(SeqCst);
        y.load(Relaxed)
    });

    let j3 = spawn(move || {
        y.store(1, Relaxed);
        fence(SeqCst);
        x.load(Relaxed)
    });

    j1.join().unwrap();
    let b = j2.join().unwrap();
    let c = j3.join().unwrap();

    assert_ne!((b, c), (0, 0));
}

fn test_corr() {
    let x = static_atomic(0);
    let y = static_atomic(0);

    let j1 = spawn(move || {
        x.store(1, Relaxed);
        x.store(2, Relaxed);
    });

    let j2 = spawn(move || {
        let r2 = x.load(Relaxed); // -------------------------------------+
        y.store(1, Release); // ---------------------+                    |
        r2 //                                        |                    |
    }); //                                           |                    |
    //                                               |synchronizes-with   |happens-before
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

    let j1 = spawn(move || {
        x.store(1, Release); // ---------------------+---------------------+
    }); //                                           |                     |
    //                                               |synchronizes-with    |
    let j2 = spawn(move || { //                      |                     |
        acquires_value(&x, 1); // <------------------+                     |
        y.store(1, Release); // ---------------------+                     |happens-before
    }); //                                           |                     |
    //                                               |synchronizes-with    |
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

    let j1 = spawn(move || {
        unsafe { *x.0 = 1 }; // -----------------------------------------+
        y.store(1, Release); // ---------------------+                   |
    }); //                                           |                   |
    //                                               |synchronizes-with  | happens-before
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

pub fn main() {
    // TODO: does this make chances of spurious success
    // "sufficiently low"? This also takes a long time to run,
    // prehaps each function should be its own test case so they
    // can be run in parallel
    for _ in 0..500 {
        test_mixed_access();
        test_load_buffering_acq_rel();
        test_message_passing();
        test_wrc();
        test_corr();
        test_rwc_syncs();
    }
}
