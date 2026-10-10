//@compile-flags: -Zmiri-compare-exchange-weak-failure-rate=0.0
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::*;

// Ensure that compare_exchange_weak never fails.
fn main() {
    let atomic = AtomicBool::new(false);
    let tries = 100;
    for _ in 0..tries {
        let cur = atomic.load(Relaxed);
        // Try (weakly) to flip the flag.
        if atomic.compare_exchange_weak(cur, !cur, Relaxed, Relaxed).is_err() {
            // We failed. Avoid panic machinery as that uses atomics/locks.
            eprintln!("compare_exchange_weak failed");
            std::process::abort();
        }
    }
}
