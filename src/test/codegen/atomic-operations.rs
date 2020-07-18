// Code generation of atomic operations.
//
// compile-flags: -O
#![crate_type = "lib"]

use std::sync::atomic::{AtomicI32, Ordering::*};

// CHECK-LABEL: @compare_exchange
#[no_mangle]
pub fn compare_exchange(a: &AtomicI32) {
    // CHECK: cmpxchg i32* %{{.*}}, i32 0, i32 10 monotonic monotonic
    let _ = a.compare_exchange(0, 10, Relaxed, Relaxed);

    // CHECK: cmpxchg i32* %{{.*}}, i32 0, i32 20 release monotonic
    let _ = a.compare_exchange(0, 20, Release, Relaxed);

    // CHECK: cmpxchg i32* %{{.*}}, i32 0, i32 30 acquire monotonic
    // CHECK: cmpxchg i32* %{{.*}}, i32 0, i32 31 acquire acquire
    let _ = a.compare_exchange(0, 30, Acquire, Relaxed);
    let _ = a.compare_exchange(0, 31, Acquire, Acquire);

    // CHECK: cmpxchg i32* %{{.*}}, i32 0, i32 40 acq_rel monotonic
    // CHECK: cmpxchg i32* %{{.*}}, i32 0, i32 41 acq_rel acquire
    let _ = a.compare_exchange(0, 40, AcqRel, Relaxed);
    let _ = a.compare_exchange(0, 41, AcqRel, Acquire);

    // CHECK: cmpxchg i32* %{{.*}}, i32 0, i32 50 seq_cst monotonic
    // CHECK: cmpxchg i32* %{{.*}}, i32 0, i32 51 seq_cst acquire
    // CHECK: cmpxchg i32* %{{.*}}, i32 0, i32 52 seq_cst seq_cst
    let _ = a.compare_exchange(0, 50, SeqCst, Relaxed);
    let _ = a.compare_exchange(0, 51, SeqCst, Acquire);
    let _ = a.compare_exchange(0, 52, SeqCst, SeqCst);
}

// CHECK-LABEL: @compare_exchange_weak
#[no_mangle]
pub fn compare_exchange_weak(w: &AtomicI32) {
    // CHECK: cmpxchg weak i32* %{{.*}}, i32 1, i32 10 monotonic monotonic
    let _ = w.compare_exchange_weak(1, 10, Relaxed, Relaxed);

    // CHECK: cmpxchg weak i32* %{{.*}}, i32 1, i32 20 release monotonic
    let _ = w.compare_exchange_weak(1, 20, Release, Relaxed);

    // CHECK: cmpxchg weak i32* %{{.*}}, i32 1, i32 30 acquire monotonic
    // CHECK: cmpxchg weak i32* %{{.*}}, i32 1, i32 31 acquire acquire
    let _ = w.compare_exchange_weak(1, 30, Acquire, Relaxed);
    let _ = w.compare_exchange_weak(1, 31, Acquire, Acquire);

    // CHECK: cmpxchg weak i32* %{{.*}}, i32 1, i32 40 acq_rel monotonic
    // CHECK: cmpxchg weak i32* %{{.*}}, i32 1, i32 41 acq_rel acquire
    let _ = w.compare_exchange_weak(1, 40, AcqRel, Relaxed);
    let _ = w.compare_exchange_weak(1, 41, AcqRel, Acquire);

    // CHECK: cmpxchg weak i32* %{{.*}}, i32 1, i32 50 seq_cst monotonic
    // CHECK: cmpxchg weak i32* %{{.*}}, i32 1, i32 51 seq_cst acquire
    // CHECK: cmpxchg weak i32* %{{.*}}, i32 1, i32 52 seq_cst seq_cst
    let _ = w.compare_exchange_weak(1, 50, SeqCst, Relaxed);
    let _ = w.compare_exchange_weak(1, 51, SeqCst, Acquire);
    let _ = w.compare_exchange_weak(1, 52, SeqCst, SeqCst);
}
