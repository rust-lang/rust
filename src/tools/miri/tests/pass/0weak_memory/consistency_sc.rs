//@compile-flags: -Zmiri-ignore-leaks -Zmiri-disable-stacked-borrows -Zmiri-disable-validation
// This test's runtime explodes if the GC interval is set to 1 (which we do in CI), so we
// override it internally back to the default frequency.
//@compile-flags: -Zmiri-provenance-gc=10000

// The following tests check whether our weak memory emulation produces
// any inconsistent execution outcomes
// This file here focuses on SC accesses and fences.

use std::sync::atomic::Ordering::*;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering, fence};
use std::thread::spawn;

// We can't create static items because we need to run each test
// multiple times
fn static_atomic(val: i32) -> &'static AtomicI32 {
    Box::leak(Box::new(AtomicI32::new(val)))
}
fn static_atomic_bool(val: bool) -> &'static AtomicBool {
    Box::leak(Box::new(AtomicBool::new(val)))
}

/// Spins until it acquires a pre-determined value.
fn spin_until_i32(loc: &AtomicI32, ord: Ordering, val: i32) -> i32 {
    while loc.load(ord) != val {
        std::hint::spin_loop();
    }
    val
}

/// Spins until it acquires a pre-determined boolean.
fn spin_until_bool(loc: &AtomicBool, ord: Ordering, val: bool) -> bool {
    while loc.load(ord) != val {
        std::hint::spin_loop();
    }
    val
}

// Test case SB taken from Repairing Sequential Consistency in C/C++11
// by Lahav et al.
// https://plv.mpi-sws.org/scfix/paper.pdf
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

// Test case by @SabrinaJewson
// https://github.com/rust-lang/miri/issues/2301#issuecomment-1221502757
// Demonstrating C++20 SC access changes
fn test_iriw_sc_rlx() {
    let x = static_atomic_bool(false);
    let y = static_atomic_bool(false);

    let a = spawn(move || x.store(true, Relaxed));
    let b = spawn(move || y.store(true, Relaxed));
    let c = spawn(move || {
        spin_until_bool(x, SeqCst, true);
        y.load(SeqCst)
    });
    let d = spawn(move || {
        spin_until_bool(y, SeqCst, true);
        x.load(SeqCst)
    });

    a.join().unwrap();
    b.join().unwrap();
    let c = c.join().unwrap();
    let d = d.join().unwrap();

    assert!(c || d);
}

// Similar to `test_iriw_sc_rlx` but with fences instead of SC accesses.
fn test_cpp20_sc_fence_fix() {
    let x = static_atomic_bool(false);
    let y = static_atomic_bool(false);

    let thread1 = spawn(|| {
        let a = x.load(Relaxed);
        fence(SeqCst);
        let b = y.load(Relaxed);
        (a, b)
    });

    let thread2 = spawn(|| {
        x.store(true, Relaxed);
    });
    let thread3 = spawn(|| {
        y.store(true, Relaxed);
    });

    let thread4 = spawn(|| {
        let c = y.load(Relaxed);
        fence(SeqCst);
        let d = x.load(Relaxed);
        (c, d)
    });

    let (a, b) = thread1.join().unwrap();
    thread2.join().unwrap();
    thread3.join().unwrap();
    let (c, d) = thread4.join().unwrap();
    let bad = a == true && b == false && c == true && d == false;
    assert!(!bad);
}

// https://plv.mpi-sws.org/scfix/paper.pdf
// 2.2 Second Problem: SC Fences are Too Weak
fn test_cpp20_rwc_syncs() {
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
        spin_until_i32(&x, Relaxed, 1);
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

    assert!((b, c) != (0, 0));
}

/// This checks that the *last* thing the SC fence does is act like a release fence.
/// See <https://github.com/rust-lang/miri/pull/4057#issuecomment-2522296601>.
/// Test by Ori Lahav.
fn test_sc_fence_release() {
    let x = static_atomic(0);
    let y = static_atomic(0);
    let z = static_atomic(0);
    let k = static_atomic(0);

    let j1 = spawn(move || {
        x.store(1, Relaxed);
        fence(SeqCst);
        k.store(1, Relaxed);
    });
    let j2 = spawn(move || {
        y.store(1, Relaxed);
        fence(SeqCst);
        z.store(1, Relaxed);
    });

    let j3 = spawn(move || {
        let kval = k.load(Acquire); // bad case: loads 1
        let yval = y.load(Relaxed); // bad case: loads 0
        (kval, yval)
    });
    let j4 = spawn(move || {
        let zval = z.load(Acquire); // bad case: loads 1
        let xval = x.load(Relaxed); // bad case: loads 0
        (zval, xval)
    });

    j1.join().unwrap();
    j2.join().unwrap();
    let (kval, yval) = j3.join().unwrap();
    let (zval, xval) = j4.join().unwrap();

    let bad = kval == 1 && yval == 0 && zval == 1 && xval == 0;
    assert!(!bad);
}

/// Test that SC fences and accesses sync correctly with each other.
/// Test by Ori Lahav.
fn test_sc_fence_access() {
    /*
        Wx1 sc
        Ry0 sc
        ||
        Wy1 rlx
        SC-fence
        Rx0 rlx
    */
    let x = static_atomic(0);
    let y = static_atomic(0);

    let j1 = spawn(move || {
        x.store(1, SeqCst);
        y.load(SeqCst)
    });
    let j2 = spawn(move || {
        y.store(1, Relaxed);
        fence(SeqCst);
        // If this sees a 0, the fence must have been *before* the x.store(1).
        x.load(Relaxed)
    });

    let yval = j1.join().unwrap();
    let xval = j2.join().unwrap();
    let bad = yval == 0 && xval == 0;
    assert!(!bad);
}

/// Test that SC fences and accesses sync correctly with each other
/// when mediated by a release-acquire pair.
/// Test by Ori Lahav (https://github.com/rust-lang/miri/pull/4057#issuecomment-2525268730).
fn test_sc_fence_access_relacq() {
    let x = static_atomic(0);
    let y = static_atomic(0);
    let z = static_atomic(0);

    let j1 = spawn(move || {
        x.store(1, SeqCst);
        y.load(SeqCst) // bad case: loads 0
    });
    let j2 = spawn(move || {
        y.store(1, Relaxed);
        z.store(1, Release)
    });
    let j3 = spawn(move || {
        let zval = z.load(Acquire); // bad case: loads 1
        // If we see 1 here, the rel-acq pair makes the fence happen after the z.store(1).
        fence(SeqCst);
        // If this sees a 0, the fence must have been *before* the x.store(1).
        let xval = x.load(Relaxed); // bad case: loads 0
        (zval, xval)
    });

    let yval = j1.join().unwrap();
    j2.join().unwrap();
    let (zval, xval) = j3.join().unwrap();
    let bad = yval == 0 && zval == 1 && xval == 0;
    assert!(!bad);
}

/// A test that involves multiple SC fences and accesses.
/// Test by Ori Lahav (https://github.com/rust-lang/miri/pull/4057#issuecomment-2525268730).
fn test_sc_multi_fence() {
    let x = static_atomic(0);
    let y = static_atomic(0);
    let z = static_atomic(0);

    let j1 = spawn(move || {
        x.store(1, SeqCst);
        y.load(SeqCst) // bad case: loads 0
    });
    let j2 = spawn(move || {
        y.store(1, Relaxed);
        // In the bad case this fence is *after* the j1 y.load, since
        // otherwise that load would pick up the 1 we just stored.
        fence(SeqCst);
        z.load(Relaxed) // bad case: loads 0
    });
    let j3 = spawn(move || {
        z.store(1, Relaxed);
    });
    let j4 = spawn(move || {
        let zval = z.load(Relaxed); // bad case: loads 1
        // In the bad case this fence is *after* the one above since
        // otherwise, the j2 load of z would load 1.
        fence(SeqCst);
        // Since that fence is in turn after the j1 y.load, our fence is
        // after the j1 x.store, which means we must pick up that store.
        let xval = x.load(Relaxed); // bad case: loads 0
        (zval, xval)
    });

    let yval = j1.join().unwrap();
    let zval1 = j2.join().unwrap();
    j3.join().unwrap();
    let (zval2, xval) = j4.join().unwrap();
    let bad = yval == 0 && zval1 == 0 && zval2 == 1 && xval == 0;
    assert!(!bad);
}

fn test_sc_relaxed() {
    /*
    y:=1 rlx
    Fence sc
    a:=x rlx
    Fence acq
    b:=z rlx // 0
    ||
    z:=1 rlx
    x:=1 sc
    c:=y sc // 0
    */

    let x = static_atomic(0);
    let y = static_atomic(0);
    let z = static_atomic(0);

    let j1 = spawn(move || {
        y.store(1, Relaxed);
        fence(SeqCst);
        // If the relaxed load here is removed, then the "bad" behavior becomes allowed
        // by C++20 (and by RC11 / scfix as well).
        let _a = x.load(Relaxed);
        fence(Acquire);
        // If we see 0 here this means in some sense we are "before" the store to z below.
        let b = z.load(Relaxed);
        b
    });
    let j2 = spawn(move || {
        z.store(1, Relaxed);
        x.store(1, SeqCst);
        // If we see 0 here, this means in some sense we are "before" the store to y above.
        let c = y.load(SeqCst);
        c
    });

    let b = j1.join().unwrap();
    let c = j2.join().unwrap();
    let bad = b == 0 && c == 0;
    assert!(!bad);
}

/// Test that coherence-ordered-before established by modification order prevents
/// an SC load from reading before the last-in-mo store.
/// Test by Joseph Perez (https://github.com/rust-lang/miri/issues/5104#issue-4644401606)
fn test_cob_due_to_mo() {
    let x = static_atomic(0);
    let y = static_atomic(0);
    x.store(0, Relaxed);
    y.store(0, Relaxed);

    let t1 = spawn(move || x.store(1, Release));
    let t2 = spawn(move || {
        let r0 = x.load(Relaxed);
        x.store(2, SeqCst);
        let r1 = y.load(SeqCst);
        (r0, r1)
    });
    let t3 = spawn(move || {
        y.store(1, SeqCst);
        let r2 = x.load(SeqCst);
        r2
    });
    let (r0, r1) = t2.join().unwrap();
    let r2 = t3.join().unwrap();
    t1.join().unwrap();
    let bad = r0 == 1 && r1 == 0 && r2 == 1;
    // This is bad because: T2's SC load of Y read 0, which means that
    // T2's SC operations all precede T3's SC operations in the SC order.
    // But in X's modification order, we must have T1's store before T2's store
    // (since T2 observed a 1). As a result, T3's X.load reading from T1's store
    // means it is coherence-ordered before and precedes in SC T2's X.store.
    // This causes a cycle in SC.
    assert!(!bad);
}

/// Variant of the test above, where the modification order is established by
/// happens-before across three threads (t1, t2, and t3).
fn test_cob_due_to_mo_hb() {
    let x = static_atomic(0);
    let y = static_atomic(0);
    let gate = static_atomic(0);
    x.store(0, Relaxed);
    y.store(0, Relaxed);
    gate.store(0, Relaxed);

    let t1 = spawn(move || x.store(1, Release));
    let t2 = spawn(move || {
        let r0 = x.load(Acquire);
        gate.store(1, Release);
        r0
    });

    let t3 = spawn(move || {
        while gate.load(Acquire) == 0 {}
        x.store(2, SeqCst);
        let r1 = y.load(SeqCst);
        r1
    });
    let t4 = spawn(move || {
        y.store(1, SeqCst);
        let r2 = x.load(SeqCst);
        r2
    });

    let r0 = t2.join().unwrap();
    let r1 = t3.join().unwrap();
    let r2 = t4.join().unwrap();
    t1.join().unwrap();
    let bad = r0 == 1 && r1 == 0 && r2 == 1;
    // This is bad because: T3's SC load of Y read 0 means that
    // T3's SC operations all precede T4's SC operations in the SC order.
    // But in X's modification order, we must have T1's store before T3's store
    // (since T1's store happens-before T3's store).
    // As a result, T4's X.load reading from T1's store means it is
    // coherence-ordered before and precedes in SC T3's X.store.
    // This causes a cycle in SC.

    assert!(!bad);
}

fn main() {
    for _ in 0..32 {
        test_sc_store_buffering();
        test_iriw_sc_rlx();
        test_cpp20_sc_fence_fix();
        test_cpp20_rwc_syncs();
        test_sc_fence_release();
        test_sc_fence_access();
        test_sc_fence_access_relacq();
        test_sc_multi_fence();
        test_sc_relaxed();
        test_cob_due_to_mo();
        test_cob_due_to_mo_hb();
    }
}
