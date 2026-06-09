// Code generation of atomic operations.
//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering::*;

// CHECK-LABEL: @compare_exchange
#[no_mangle]
pub fn compare_exchange(a: &AtomicI32) {
    // CHECK: cmpxchg ptr %{{.*}}, i32 0, i32 10 monotonic monotonic
    // CHECK: cmpxchg ptr %{{.*}}, i32 0, i32 11 monotonic acquire
    // CHECK: cmpxchg ptr %{{.*}}, i32 0, i32 12 monotonic seq_cst
    let _ = a.compare_exchange(0, 10, Relaxed, Relaxed);
    let _ = a.compare_exchange(0, 11, Relaxed, Acquire);
    let _ = a.compare_exchange(0, 12, Relaxed, SeqCst);

    // CHECK: cmpxchg ptr %{{.*}}, i32 0, i32 20 release monotonic
    // CHECK: cmpxchg ptr %{{.*}}, i32 0, i32 21 release acquire
    // CHECK: cmpxchg ptr %{{.*}}, i32 0, i32 22 release seq_cst
    let _ = a.compare_exchange(0, 20, Release, Relaxed);
    let _ = a.compare_exchange(0, 21, Release, Acquire);
    let _ = a.compare_exchange(0, 22, Release, SeqCst);

    // CHECK: cmpxchg ptr %{{.*}}, i32 0, i32 30 acquire monotonic
    // CHECK: cmpxchg ptr %{{.*}}, i32 0, i32 31 acquire acquire
    // CHECK: cmpxchg ptr %{{.*}}, i32 0, i32 32 acquire seq_cst
    let _ = a.compare_exchange(0, 30, Acquire, Relaxed);
    let _ = a.compare_exchange(0, 31, Acquire, Acquire);
    let _ = a.compare_exchange(0, 32, Acquire, SeqCst);

    // CHECK: cmpxchg ptr %{{.*}}, i32 0, i32 40 acq_rel monotonic
    // CHECK: cmpxchg ptr %{{.*}}, i32 0, i32 41 acq_rel acquire
    // CHECK: cmpxchg ptr %{{.*}}, i32 0, i32 42 acq_rel seq_cst
    let _ = a.compare_exchange(0, 40, AcqRel, Relaxed);
    let _ = a.compare_exchange(0, 41, AcqRel, Acquire);
    let _ = a.compare_exchange(0, 42, AcqRel, SeqCst);

    // CHECK: cmpxchg ptr %{{.*}}, i32 0, i32 50 seq_cst monotonic
    // CHECK: cmpxchg ptr %{{.*}}, i32 0, i32 51 seq_cst acquire
    // CHECK: cmpxchg ptr %{{.*}}, i32 0, i32 52 seq_cst seq_cst
    let _ = a.compare_exchange(0, 50, SeqCst, Relaxed);
    let _ = a.compare_exchange(0, 51, SeqCst, Acquire);
    let _ = a.compare_exchange(0, 52, SeqCst, SeqCst);
}

// CHECK-LABEL: @compare_exchange_weak
#[no_mangle]
pub fn compare_exchange_weak(w: &AtomicI32) {
    // CHECK: cmpxchg weak ptr %{{.*}}, i32 1, i32 10 monotonic monotonic
    // CHECK: cmpxchg weak ptr %{{.*}}, i32 1, i32 11 monotonic acquire
    // CHECK: cmpxchg weak ptr %{{.*}}, i32 1, i32 12 monotonic seq_cst
    let _ = w.compare_exchange_weak(1, 10, Relaxed, Relaxed);
    let _ = w.compare_exchange_weak(1, 11, Relaxed, Acquire);
    let _ = w.compare_exchange_weak(1, 12, Relaxed, SeqCst);

    // CHECK: cmpxchg weak ptr %{{.*}}, i32 1, i32 20 release monotonic
    // CHECK: cmpxchg weak ptr %{{.*}}, i32 1, i32 21 release acquire
    // CHECK: cmpxchg weak ptr %{{.*}}, i32 1, i32 22 release seq_cst
    let _ = w.compare_exchange_weak(1, 20, Release, Relaxed);
    let _ = w.compare_exchange_weak(1, 21, Release, Acquire);
    let _ = w.compare_exchange_weak(1, 22, Release, SeqCst);

    // CHECK: cmpxchg weak ptr %{{.*}}, i32 1, i32 30 acquire monotonic
    // CHECK: cmpxchg weak ptr %{{.*}}, i32 1, i32 31 acquire acquire
    // CHECK: cmpxchg weak ptr %{{.*}}, i32 1, i32 32 acquire seq_cst
    let _ = w.compare_exchange_weak(1, 30, Acquire, Relaxed);
    let _ = w.compare_exchange_weak(1, 31, Acquire, Acquire);
    let _ = w.compare_exchange_weak(1, 32, Acquire, SeqCst);

    // CHECK: cmpxchg weak ptr %{{.*}}, i32 1, i32 40 acq_rel monotonic
    // CHECK: cmpxchg weak ptr %{{.*}}, i32 1, i32 41 acq_rel acquire
    // CHECK: cmpxchg weak ptr %{{.*}}, i32 1, i32 42 acq_rel seq_cst
    let _ = w.compare_exchange_weak(1, 40, AcqRel, Relaxed);
    let _ = w.compare_exchange_weak(1, 41, AcqRel, Acquire);
    let _ = w.compare_exchange_weak(1, 42, AcqRel, SeqCst);

    // CHECK: cmpxchg weak ptr %{{.*}}, i32 1, i32 50 seq_cst monotonic
    // CHECK: cmpxchg weak ptr %{{.*}}, i32 1, i32 51 seq_cst acquire
    // CHECK: cmpxchg weak ptr %{{.*}}, i32 1, i32 52 seq_cst seq_cst
    let _ = w.compare_exchange_weak(1, 50, SeqCst, Relaxed);
    let _ = w.compare_exchange_weak(1, 51, SeqCst, Acquire);
    let _ = w.compare_exchange_weak(1, 52, SeqCst, SeqCst);
}
